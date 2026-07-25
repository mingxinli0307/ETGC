#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SEEDS_AB = [42, 43, 44, 45, 46]
SEEDS_CD = [42, 43, 44]
FOREST_SEED = 20260725
PREFERRED_DATASETS = ["school", "patent", "dblp"]


def stable_run_key(config):
    keys = [
        "dataset",
        "init_mode",
        "lloyd_iters",
        "penalty_type",
        "penalty_weight",
        "warmup_epochs",
        "lambda_prox",
        "model_seed",
        "prototype_seed",
        "forest_seed",
        "direct_kmeans_eval",
        "init_only",
        "epoch",
    ]
    payload = {key: config.get(key) for key in keys}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    stem = "_".join(
        [
            str(config.get("dataset", "")),
            str(config.get("init_mode", "")),
            str(config.get("penalty_type", "")),
            str(config.get("penalty_weight", "")).replace(".", "p"),
            "w" + str(config.get("warmup_epochs", 0)),
            "s" + str(config.get("model_seed", "")),
        ]
    )
    return f"{stem}_{digest}"


def is_success_result(path):
    result = Path(path) / "result.json"
    if not result.exists():
        return False
    try:
        return json.loads(result.read_text(encoding="utf-8")).get("status") == "success"
    except Exception:
        return False


def resolve_dataset(root, name):
    ds_root = Path(root) / "dataset"
    direct = ds_root / name
    if (direct / f"{name}.txt").exists() and (direct / "node2label.txt").exists():
        return name
    if ds_root.exists():
        for child in ds_root.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                if (child / f"{child.name}.txt").exists() and (child / "node2label.txt").exists():
                    return child.name
    return name


def load_source_config(root, dataset, seed, config_source_dir=None):
    cfg = {}
    if config_source_dir:
        path = Path(config_source_dir) / f"D1_trace_only_seed{seed}" / "config.json"
        if path.exists():
            cfg = json.loads(path.read_text(encoding="utf-8"))
    defaults = {
        "directed": 0,
        "batch_size": 512,
        "learning_rate": 1e-4,
        "edge_dim": 128,
        "time_dim": 32,
        "edge_hidden_dim": 128,
        "cluster_hidden_dim": 64,
        "alpha": 0.2,
        "T": 4,
        "beta": 5.0,
        "edge_neighbor_k": -1,
        "edge_ppr_topk": -1,
        "edge_ppr_method": "temporal_state_forest",
        "forest_samples": 5,
        "global_q_chunk_size": 8192,
        "global_ncut_row_block_size": 65536,
        "node_emb_lr": 1e-5,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    cfg["dataset"] = resolve_dataset(root, dataset)
    cfg["seed"] = int(seed)
    return cfg


def base_run_config(root, out_dir, group, name, dataset, seed, config_source_dir=None):
    src = load_source_config(root, dataset, seed, config_source_dir)
    return {
        "group": group,
        "name": name,
        "dataset": src["dataset"],
        "model_seed": int(seed),
        "prototype_seed": int(seed),
        "forest_seed": FOREST_SEED,
        "directed": src["directed"],
        "batch_size": src["batch_size"],
        "learning_rate": src["learning_rate"],
        "edge_dim": src["edge_dim"],
        "time_dim": src["time_dim"],
        "edge_hidden_dim": src["edge_hidden_dim"],
        "cluster_hidden_dim": src["cluster_hidden_dim"],
        "alpha": src["alpha"],
        "T": src["T"],
        "beta": src["beta"],
        "edge_neighbor_k": src["edge_neighbor_k"],
        "edge_ppr_topk": src["edge_ppr_topk"],
        "edge_ppr_method": src["edge_ppr_method"],
        "forest_samples": src["forest_samples"],
        "global_q_chunk_size": src["global_q_chunk_size"],
        "global_ncut_row_block_size": src["global_ncut_row_block_size"],
        "node_emb_lr": src["node_emb_lr"],
        "cluster_input_norm": "layernorm",
        "cluster_output_bias_mode": "none",
        "cluster_init_mode": "random",
        "init_mode": "random",
        "lloyd_iters": 0,
        "penalty_type": "orth",
        "penalty_weight": 1.0,
        "lambda_prox": 0.0,
        "lambda_edge_ncut": 0.5,
        "lambda_proj": 0.0,
        "lambda_bal": 0.0,
        "node_emb_mode": "frozen",
        "warmup_epochs": 0,
        "epoch": 30,
        "direct_kmeans_eval": 0,
        "init_only": 0,
        "prototype_sample_size": 20000,
        "data_root": str(Path(root) / "dataset"),
        "cache_dir": str(Path(root) / "cache"),
        "emb_root": str(Path(root) / "emb"),
        "pretrain_emb_dir": str(Path(root) / "pretrain"),
    }


def make_plan(root, out_dir, groups, config_source_dir=None):
    selected = set(groups)
    if "all" in selected:
        selected = {"A", "B", "C", "D"}
    plan = []
    if "A" in selected or "B" in selected:
        dataset = "school"
        if "A" in selected:
            for seed in SEEDS_AB:
                cfg = base_run_config(root, out_dir, "A", "A0_random_etgc", dataset, seed, config_source_dir)
                cfg.update({"cluster_init_mode": "random", "init_mode": "random"})
                plan.append(cfg)
                cfg = base_run_config(root, out_dir, "A", "A1_direct_kmeans", dataset, seed, config_source_dir)
                cfg.update({"direct_kmeans_eval": 1, "epoch": 0, "prototype_lloyd_iters": 10, "lloyd_iters": 10})
                plan.append(cfg)
                cfg = base_run_config(root, out_dir, "A", "A2_prototype_init_only", dataset, seed, config_source_dir)
                cfg.update({"cluster_init_mode": "prototype", "init_mode": "prototype", "prototype_lloyd_iters": 10, "lloyd_iters": 10, "init_only": 1, "epoch": 0})
                plan.append(cfg)
                cfg = base_run_config(root, out_dir, "A", "A3_prototype_etgc", dataset, seed, config_source_dir)
                cfg.update({"cluster_init_mode": "prototype", "init_mode": "prototype", "prototype_lloyd_iters": 10, "lloyd_iters": 10})
                plan.append(cfg)
        if "B" in selected:
            b_configs = [
                ("B1_random_orthogonal", "random_orthogonal", 0),
                ("B2_random_event", "random_event", 0),
                ("B3_kmeans_plus_plus", "kmeans_plus_plus", 0),
                ("B4_kmeans_lloyd1", "prototype", 1),
            ]
            for seed in SEEDS_AB:
                for name, init_mode, lloyd in b_configs:
                    cfg = base_run_config(root, out_dir, "B", name, dataset, seed, config_source_dir)
                    cfg.update({"cluster_init_mode": init_mode, "init_mode": init_mode, "prototype_lloyd_iters": lloyd, "lloyd_iters": lloyd})
                    plan.append(cfg)
    if "C" in selected:
        c_order = [
            ("school", "R"),
            ("school", "K"),
            ("patent", "R"),
            ("patent", "K"),
            ("dblp", "R"),
            ("dblp", "K"),
        ]
        penalties = [("orth", 1.0), ("orthqa", 0.1), ("orthqa", 0.5), ("orthqa", 1.0), ("orthqa", 2.0)]
        for dataset, path in c_order:
            for penalty_index, (penalty, weight) in enumerate(penalties):
                for seed in SEEDS_CD:
                    cfg = base_run_config(root, out_dir, "C", f"C-{path}{penalty_index}_{penalty}_{weight}", dataset, seed, config_source_dir)
                    if path == "R":
                        cfg.update({"cluster_init_mode": "random_orthogonal", "init_mode": "random_orthogonal", "lloyd_iters": 0})
                    else:
                        cfg.update({"cluster_init_mode": "prototype", "init_mode": "prototype", "prototype_lloyd_iters": 10, "lloyd_iters": 10})
                    cfg.update({"penalty_type": penalty, "penalty_weight": weight})
                    plan.append(cfg)
    if "D" in selected:
        d_configs = [
            ("D0_warm0_orth", 0, "orth"),
            ("D1_warm5_orth", 5, "orth"),
            ("D2_warm10_orth", 10, "orth"),
            ("D3_warm20_orth", 20, "orth"),
            ("D4_warm5_orthqa", 5, "orthqa"),
            ("D5_warm10_orthqa", 10, "orthqa"),
            ("D6_warm20_orthqa", 20, "orthqa"),
        ]
        for seed in SEEDS_CD:
            for name, warmup, penalty in d_configs:
                cfg = base_run_config(root, out_dir, "D", name, "school", seed, config_source_dir)
                cfg.update(
                    {
                        "cluster_init_mode": "random_orthogonal",
                        "init_mode": "random_orthogonal",
                        "lambda_prox": 1.0,
                        "warmup_epochs": warmup,
                        "epoch": warmup + 30,
                        "penalty_type": penalty,
                        "penalty_weight": 1.0,
                    }
                )
                plan.append(cfg)
    return plan


def group_dir(out_dir, cfg):
    group = cfg["group"]
    if group == "A":
        return Path(out_dir) / "group_A_kmeans_contribution" / cfg["name"] / f"seed_{cfg['model_seed']}"
    if group == "B":
        return Path(out_dir) / "group_B_initialization" / cfg["name"] / f"seed_{cfg['model_seed']}"
    if group == "C":
        return Path(out_dir) / "group_C_orthqa" / cfg["dataset"].lower() / cfg["name"] / f"seed_{cfg['model_seed']}"
    if group == "D":
        return Path(out_dir) / "group_D_warmup" / cfg["name"] / f"seed_{cfg['model_seed']}"
    return Path(out_dir) / "runs" / stable_run_key(cfg)


def command_for(root, cfg, run_dir, python_bin, device, q_chunk=None, row_block=None):
    q_chunk = int(q_chunk or cfg["global_q_chunk_size"])
    row_block = int(row_block or cfg["global_ncut_row_block_size"])
    args = {
        "dataset": cfg["dataset"],
        "directed": cfg["directed"],
        "device": device,
        "seed": cfg["model_seed"],
        "model_seed": cfg["model_seed"],
        "prototype_seed": cfg["prototype_seed"],
        "forest_seed": cfg["forest_seed"],
        "data_root": cfg["data_root"],
        "emb_root": cfg["emb_root"],
        "pretrain_emb_dir": cfg["pretrain_emb_dir"],
        "cache_dir": cfg["cache_dir"],
        "batch_size": cfg["batch_size"],
        "epoch": cfg["epoch"],
        "learning_rate": cfg["learning_rate"],
        "edge_dim": cfg["edge_dim"],
        "time_dim": cfg["time_dim"],
        "edge_hidden_dim": cfg["edge_hidden_dim"],
        "cluster_hidden_dim": cfg["cluster_hidden_dim"],
        "alpha": cfg["alpha"],
        "T": cfg["T"],
        "beta": cfg["beta"],
        "edge_neighbor_k": cfg["edge_neighbor_k"],
        "edge_ppr_topk": cfg["edge_ppr_topk"],
        "edge_ppr_method": cfg["edge_ppr_method"],
        "forest_samples": cfg["forest_samples"],
        "ncut_scope": "global",
        "cluster_loss_type": "trace_mincut",
        "orth_type": cfg["penalty_type"],
        "lambda_orth": cfg["penalty_weight"],
        "global_q_chunk_size": q_chunk,
        "global_ncut_row_block_size": row_block,
        "global_warmup_epochs": cfg["warmup_epochs"],
        "quiet": 1,
        "lambda_prox": cfg["lambda_prox"],
        "lambda_edge_ncut": cfg["lambda_edge_ncut"],
        "lambda_proj": cfg["lambda_proj"],
        "lambda_bal": cfg["lambda_bal"],
        "node_emb_mode": cfg["node_emb_mode"],
        "node_emb_lr": cfg["node_emb_lr"],
        "cluster_output_bias_mode": cfg["cluster_output_bias_mode"],
        "cluster_input_norm": cfg["cluster_input_norm"],
        "cluster_init_mode": cfg["cluster_init_mode"],
        "prototype_sample_size": cfg["prototype_sample_size"],
        "prototype_lloyd_iters": cfg.get("prototype_lloyd_iters", cfg.get("lloyd_iters", 0)),
        "direct_kmeans_eval": cfg["direct_kmeans_eval"],
        "init_only": cfg["init_only"],
        "overnight_diagnostic": 1,
        "uniform_collapse_diagnostic": 1,
        "diagnostic_only_first_epoch": 0,
        "diagnostic_epochs": "1,5,10,20,30",
        "output_dir": str(run_dir),
        "diagnostic_output_dir": str(run_dir),
        "save_embeddings": 0,
    }
    cmd = [python_bin, "edge_main.py"]
    for key, value in args.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def write_failure(run_dir, cfg, error_type, message, exit_code, runtime):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    payload = dict(cfg)
    payload.update(
        {
            "status": "failed",
            "error_type": error_type,
            "error_message": str(message)[:1000],
            "traceback_tail": str(message)[-3000:],
            "exit_code": int(exit_code),
            "runtime_seconds": float(runtime),
        }
    )
    (Path(run_dir) / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run_one(root, cfg, out_dir, python_bin, device, resume):
    run_dir = group_dir(out_dir, cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and is_success_result(run_dir):
        return "skipped", run_dir
    q_values = [8192, 4096, 2048, 1024]
    q_values = [x for x in q_values if x <= int(cfg["global_q_chunk_size"])] or [int(cfg["global_q_chunk_size"])]
    started = time.time()
    for attempt, q_chunk in enumerate(q_values[:4], start=1):
        row_block = min(int(cfg["global_ncut_row_block_size"]), q_chunk * 8)
        cmd = command_for(root, cfg, run_dir, python_bin, device, q_chunk=q_chunk, row_block=row_block)
        with (run_dir / "train.log").open("a", encoding="utf-8") as log:
            log.write(
                f"[START] group={cfg['group']} dataset={cfg['dataset']} init={cfg['init_mode']} "
                f"penalty={cfg['penalty_type']} weight={cfg['penalty_weight']} warmup={cfg['warmup_epochs']} "
                f"model_seed={cfg['model_seed']} prototype_seed={cfg['prototype_seed']} forest_seed={cfg['forest_seed']} "
                f"attempt={attempt} q_chunk={q_chunk} row_block={row_block}\n"
            )
            proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"[DONE] exit_code={proc.returncode} runtime={time.time() - started:.6f}\n")
        if proc.returncode == 0:
            return "success", run_dir
        tail = (run_dir / "train.log").read_text(encoding="utf-8", errors="ignore")[-4000:]
        if "out of memory" not in tail.lower() and "cuda oom" not in tail.lower():
            write_failure(run_dir, cfg, "runtime_error", tail, proc.returncode, time.time() - started)
            return "failed", run_dir
    tail = (run_dir / "train.log").read_text(encoding="utf-8", errors="ignore")[-4000:]
    write_failure(run_dir, cfg, "oom", tail, 1, time.time() - started)
    return "failed", run_dir


def finite(value, default=math.nan):
    try:
        if value in ("", None):
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as reader:
        return list(csv.DictReader(reader))


def latest_metric(run_dir):
    rows = read_csv_rows(Path(run_dir) / "metrics.csv")
    return rows[-1] if rows else {}


def diagnostic_stage(run_dir, stage):
    rows = read_csv_rows(Path(run_dir) / "diagnostic_summary.csv")
    for row in rows:
        if row.get("stage") == stage:
            return row
    return {}


def collect_master(out_dir):
    rows = []
    for result_path in Path(out_dir).rglob("result.json"):
        run_dir = result_path.parent
        result = read_json(result_path)
        cfg = read_json(run_dir / "config.json")
        metric = latest_metric(run_dir)
        final_stage = diagnostic_stage(run_dir, "final_epoch")
        row = {
            "run_dir": str(run_dir),
            "status": result.get("status", "missing"),
            "group": cfg.get("group", result.get("group", "")),
            "name": cfg.get("name", result.get("name", "")),
            "dataset": result.get("dataset", cfg.get("dataset", "")),
            "model_seed": result.get("model_seed", cfg.get("model_seed", cfg.get("seed", ""))),
            "prototype_seed": result.get("prototype_seed", cfg.get("prototype_seed", "")),
            "forest_seed": result.get("forest_seed", cfg.get("forest_seed", "")),
            "init_mode": cfg.get("cluster_init_mode", result.get("init_mode", "")),
            "lloyd_iters": cfg.get("prototype_lloyd_iters", result.get("lloyd_iters", "")),
            "penalty_type": cfg.get("orth_type", result.get("penalty_type", "")),
            "penalty_weight": cfg.get("lambda_orth", result.get("penalty_weight", "")),
            "warmup_epochs": cfg.get("global_warmup_epochs", result.get("warmup_epochs", "")),
            "MacroF1_best": (result.get("best_metrics") or {}).get("Macro_F1", ""),
            "MacroF1_final": (result.get("final_metrics") or {}).get("Macro_F1", metric.get("Macro_F1", "")),
            "NMI_final": (result.get("final_metrics") or {}).get("NMI", metric.get("NMI", "")),
            "ARI_final": (result.get("final_metrics") or {}).get("ARI", metric.get("ARI", "")),
            "ACC_final": (result.get("final_metrics") or {}).get("ACC", metric.get("ACC", "")),
            "runtime_seconds": result.get("runtime_seconds", ""),
            "q_rank1_energy_final": final_stage.get("q_rank1_energy_ratio", metric.get("q_rank1_energy_ratio", "")),
            "q_centered_energy_final": final_stage.get("q_centered_energy", metric.get("q_centered_energy", "")),
            "active_node_clusters_final": final_stage.get("num_active_node_clusters", metric.get("node_hard_active_clusters", "")),
            "largest_node_ratio_final": final_stage.get("largest_node_cluster_ratio", metric.get("node_hard_largest_ratio", "")),
            "cluster_volume_cv_final": final_stage.get("cluster_volume_coefficient_of_variation", metric.get("cluster_volume_coefficient_of_variation", "")),
            "hard_cluster_volume_cv_final": final_stage.get("hard_cluster_volume_cv", metric.get("hard_cluster_volume_cv", "")),
            "error_message": result.get("error_message", ""),
        }
        changes = result.get("training_change_metrics") or {}
        row.update(changes)
        rows.append(row)
    return rows


def write_csv(path, fieldnames, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as writer:
        w = csv.DictWriter(writer, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key, "") for key in fieldnames})


def mean_std(values):
    vals = [finite(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return math.nan, math.nan
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, var ** 0.5


def summarize(out_dir):
    out_dir = Path(out_dir)
    master = collect_master(out_dir)
    master_fields = sorted({key for row in master for key in row})
    write_csv(out_dir / "master_runs.csv", master_fields, master)
    failed = [row for row in master if row.get("status") != "success"]
    write_csv(out_dir / "failed_runs.csv", master_fields, failed)

    a_rows = [r for r in master if "/group_A_kmeans_contribution/" in r.get("run_dir", "")]
    write_csv(
        out_dir / "a_kmeans_contribution.csv",
        master_fields,
        a_rows,
    )
    b_rows = [r for r in master if "/group_B_initialization/" in r.get("run_dir", "")]
    write_csv(out_dir / "b_initialization_comparison.csv", master_fields, b_rows)

    c_groups = {}
    for row in master:
        if "/group_C_orthqa/" not in row.get("run_dir", ""):
            continue
        key = (row.get("dataset", ""), row.get("init_mode", ""), row.get("penalty_type", ""), str(row.get("penalty_weight", "")))
        c_groups.setdefault(key, []).append(row)
    c_rows = []
    for (dataset, init_mode, penalty, weight), rows in sorted(c_groups.items()):
        best_m, best_s = mean_std([r.get("MacroF1_best") for r in rows])
        final_m, final_s = mean_std([r.get("MacroF1_final") for r in rows])
        rank_m, rank_s = mean_std([r.get("q_rank1_energy_final") for r in rows])
        cv_m, cv_s = mean_std([r.get("cluster_volume_cv_final") for r in rows])
        collapse = 0
        for row in rows:
            if finite(row.get("active_node_clusters_final"), 999) <= 1:
                collapse += 1
            elif finite(row.get("largest_node_ratio_final"), 0) >= 0.95:
                collapse += 1
            elif finite(row.get("q_rank1_energy_final"), 0) >= 0.9999 and finite(row.get("q_centered_energy_final"), 1e9) < 1.0:
                collapse += 1
        c_rows.append(
            {
                "dataset": dataset,
                "init_path": "kmeans" if init_mode == "prototype" else "random_orthogonal",
                "penalty_type": penalty,
                "penalty_weight": weight,
                "MacroF1_best_mean": best_m,
                "MacroF1_best_std": best_s,
                "MacroF1_final_mean": final_m,
                "MacroF1_final_std": final_s,
                "rank1_energy_final_mean": rank_m,
                "rank1_energy_final_std": rank_s,
                "cluster_volume_cv_final_mean": cv_m,
                "cluster_volume_cv_final_std": cv_s,
                "collapse_run_count": collapse,
            }
        )
    write_csv(out_dir / "c_orthqa_summary.csv", sorted({k for r in c_rows for k in r}), c_rows)

    d_rows = [r for r in master if "/group_D_warmup/" in r.get("run_dir", "")]
    write_csv(out_dir / "d_warmup_summary.csv", master_fields, d_rows)

    lines = [
        "# ETGC KMeans Contribution and Orthqa Overnight Analysis",
        "",
        "This report is generated from completed result.json, metrics.csv, and diagnostic.json files.",
        "It does not select configurations using labels beyond the fixed clustering metrics already logged by each run.",
        "",
        "## Current Status",
        f"- total result files: {len(master)}",
        f"- failed runs: {len(failed)}",
        "",
        "## Required Questions",
        "- A group: compare direct KMeans, prototype initialization, prototype training, and random ETGC using a_kmeans_contribution.csv.",
        "- B group: compare random orthogonal, random event, KMeans++ only, one Lloyd step, and full KMeans using b_initialization_comparison.csv.",
        "- C group: compare orth and orthqa by dataset/init path/weight using c_orthqa_summary.csv.",
        "- D group: compare proximity warm-up schedules using d_warmup_summary.csv.",
        "",
        "Final conclusions should be based only after all planned runs complete.",
    ]
    (out_dir / "final_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"master": len(master), "failed": len(failed)}


def prewarm_cache(root, out_dir, python_bin, device, config_source_dir):
    cache_info = {}
    for dataset in PREFERRED_DATASETS:
        cfg = base_run_config(root, out_dir, "cache", f"cache_{dataset}", dataset, 42, config_source_dir)
        cfg.update({"init_only": 1, "epoch": 0, "cluster_init_mode": "random"})
        run_dir = Path(out_dir) / "code_info" / "cache_prewarm" / cfg["dataset"]
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = command_for(root, cfg, run_dir, python_bin, device)
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        cache_info[cfg["dataset"]] = {"exit_code": proc.returncode, "run_dir": str(run_dir)}
    (Path(out_dir) / "code_info" / "cache_info.json").write_text(
        json.dumps(cache_info, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_queue(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.prewarm:
        prewarm_cache(args.root, out_dir, args.python, args.device, args.config_source_dir)
    plan = make_plan(args.root, out_dir, args.groups, args.config_source_dir)
    seen_success = {}
    for cfg in plan:
        sig = stable_run_key(cfg)
        if sig in seen_success and is_success_result(seen_success[sig]):
            run_dir = group_dir(out_dir, cfg)
            run_dir.mkdir(parents=True, exist_ok=True)
            source = Path(seen_success[sig])
            for name in ["config.json", "metrics.csv", "diagnostic.json", "diagnostic_summary.csv", "train.log"]:
                src = source / name
                if src.exists():
                    shutil.copy2(src, run_dir / name)
            result = read_json(source / "result.json")
            result.update({"reused_from": str(source), **cfg})
            (run_dir / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            continue
        status, run_dir = run_one(args.root, cfg, out_dir, args.python, args.device, args.resume)
        if status in {"success", "skipped"}:
            seen_success[sig] = str(run_dir)
        summarize(out_dir)
    return summarize(out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Run and summarize ETGC KMeans/orthqa overnight experiments.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config-source-dir", default="")
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run:
        summary = run_queue(args)
    else:
        summary = summarize(args.out_dir)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
