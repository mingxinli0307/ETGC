#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summarize_direct_node_time_all_datasets import (  # noqa: E402
    discover_datasets,
    finite,
    mean_std,
    read_csv_rows,
    read_json,
    resolve_node2vec_path,
    write_csv,
    write_failure,
    write_json,
)


SEEDS = [42, 43, 44]
FOREST_SEED = 20260725
REPRESENTATIVE_DATASETS = ["school", "brain", "arxivai", "dblp"]
CONFIGS = {
    "N0": {
        "name": "N0_direct_baseline",
        "direct_time_scale": 1.0,
        "node_emb_mode": "full",
        "node_emb_lr": 1e-4,
        "prox_similarity_mode": "event_dot",
        "lambda_prox": 1.0,
        "lambda_node_anchor": 0.0,
    },
    "N1": {
        "name": "N1_time_scale_0_5",
        "direct_time_scale": 0.5,
        "node_emb_mode": "full",
        "node_emb_lr": 1e-4,
        "prox_similarity_mode": "event_dot",
        "lambda_prox": 1.0,
        "lambda_node_anchor": 0.0,
    },
    "N2": {
        "name": "N2_time_scale_0_25",
        "direct_time_scale": 0.25,
        "node_emb_mode": "full",
        "node_emb_lr": 1e-4,
        "prox_similarity_mode": "event_dot",
        "lambda_prox": 1.0,
        "lambda_node_anchor": 0.0,
    },
    "N3": {
        "name": "N3_time_scale_0_1",
        "direct_time_scale": 0.1,
        "node_emb_mode": "full",
        "node_emb_lr": 1e-4,
        "prox_similarity_mode": "event_dot",
        "lambda_prox": 1.0,
        "lambda_node_anchor": 0.0,
    },
    "N4": {
        "name": "N4_time_scale_smalllr_0_1",
        "direct_time_scale": 0.25,
        "node_emb_mode": "small_lr",
        "node_emb_lr": 1e-5,
        "prox_similarity_mode": "event_dot",
        "lambda_prox": 1.0,
        "lambda_node_anchor": 0.0,
    },
    "N5": {
        "name": "N5_time_scale_smalllr_0_01",
        "direct_time_scale": 0.25,
        "node_emb_mode": "small_lr",
        "node_emb_lr": 1e-6,
        "prox_similarity_mode": "event_dot",
        "lambda_prox": 1.0,
        "lambda_node_anchor": 0.0,
    },
    "N6": {
        "name": "N6_smalllr_weak_anchor",
        "direct_time_scale": 0.25,
        "node_emb_mode": "small_lr",
        "node_emb_lr": 1e-5,
        "prox_similarity_mode": "event_dot",
        "lambda_prox": 1.0,
        "lambda_node_anchor": 0.001,
    },
    "N7": {
        "name": "N7_smalllr_stronger_anchor",
        "direct_time_scale": 0.25,
        "node_emb_mode": "small_lr",
        "node_emb_lr": 1e-5,
        "prox_similarity_mode": "event_dot",
        "lambda_prox": 1.0,
        "lambda_node_anchor": 0.01,
    },
    "N8": {
        "name": "N8_role_aware",
        "direct_time_scale": 0.25,
        "node_emb_mode": "small_lr",
        "node_emb_lr": 1e-5,
        "prox_similarity_mode": "role_aware",
        "lambda_prox": 1.0,
        "lambda_node_anchor": 0.01,
    },
    "N9": {
        "name": "N9_role_aware_lambda_prox_0_1",
        "direct_time_scale": 0.25,
        "node_emb_mode": "small_lr",
        "node_emb_lr": 1e-5,
        "prox_similarity_mode": "role_aware",
        "lambda_prox": 0.1,
        "lambda_node_anchor": 0.01,
    },
    "N10": {
        "name": "N10_role_aware_lambda_prox_0_01",
        "direct_time_scale": 0.25,
        "node_emb_mode": "small_lr",
        "node_emb_lr": 1e-5,
        "prox_similarity_mode": "role_aware",
        "lambda_prox": 0.01,
        "lambda_node_anchor": 0.01,
    },
}

COMMON_CONFIG = {
    "directed": 0,
    "batch_size": 512,
    "epoch": 30,
    "learning_rate": 1e-4,
    "edge_dim": 128,
    "time_dim": 32,
    "edge_hidden_dim": 128,
    "cluster_hidden_dim": 64,
    "edge_encoder_mode": "direct_node_time",
    "alpha": 0.2,
    "T": 4,
    "beta": 5.0,
    "edge_neighbor_k": -1,
    "edge_ppr_topk": -1,
    "edge_ppr_method": "temporal_state_forest",
    "forest_samples": 50,
    "global_q_chunk_size": 8192,
    "global_ncut_row_block_size": 65536,
    "global_warmup_epochs": 0,
    "lambda_edge_ncut": 0.5,
    "lambda_proj": 0.0,
    "lambda_bal": 0.0,
    "orth_type": "orth",
    "lambda_orth": 1.0,
    "cluster_output_bias_mode": "none",
    "cluster_input_norm": "layernorm",
    "cluster_init_mode": "random_orthogonal",
    "prototype_sample_size": 20000,
    "prototype_lloyd_iters": 0,
    "require_pretrained_node2vec": 1,
    "prox_role_ss_weight": 0.25,
    "prox_role_dd_weight": 0.25,
    "prox_role_ds_weight": 1.0,
    "prox_role_sd_weight": 0.0,
    "prox_role_time_weight": 0.25,
    "prox_temperature": 0.2,
}


def stable_config_checksum(cfg: dict) -> str:
    keys = [
        "dataset",
        "config",
        "direct_time_scale",
        "node_emb_mode",
        "node_emb_lr",
        "lambda_node_anchor",
        "prox_similarity_mode",
        "prox_role_ss_weight",
        "prox_role_dd_weight",
        "prox_role_ds_weight",
        "prox_role_sd_weight",
        "prox_role_time_weight",
        "prox_temperature",
        "lambda_prox",
        "orth_type",
        "lambda_orth",
        "model_seed",
        "forest_seed",
        "epoch",
        "learning_rate",
        "edge_ppr_method",
        "edge_ppr_topk",
        "forest_samples",
        "edge_neighbor_k",
        "cluster_init_mode",
        "cluster_input_norm",
        "cluster_output_bias_mode",
        "require_pretrained_node2vec",
    ]
    payload = {key: cfg.get(key) for key in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def make_run_config(dataset: str, config_id: str, seed: int, root: Path) -> dict:
    cfg = dict(COMMON_CONFIG)
    cfg.update(CONFIGS[config_id])
    cfg.update(
        {
            "dataset": dataset,
            "config": config_id,
            "model_seed": int(seed),
            "prototype_seed": int(seed),
            "forest_seed": FOREST_SEED,
            "data_root": str(root / "dataset"),
            "emb_root": str(root / "emb"),
            "pretrain_emb_dir": str(root / "pretrain"),
            "cache_dir": str(root / "cache"),
        }
    )
    cfg["config_checksum"] = stable_config_checksum(cfg)
    return cfg


def run_dir_for(out_dir: Path, cfg: dict) -> Path:
    return out_dir / str(cfg["dataset"]).lower() / str(cfg["config"]) / f"seed_{cfg['model_seed']}"


def is_success_result(run_dir: Path, checksum: str) -> bool:
    result = read_json(run_dir / "result.json")
    return result.get("status") == "success" and result.get("config_checksum") == checksum


def phase_dataset_order(inventory: list, phase: str) -> list:
    usable = [row for row in inventory if bool(row.get("usable"))]
    usable.sort(key=lambda x: int(x.get("event_count") or 10**18))
    by_name = {str(row.get("dataset_name")).lower(): row for row in usable}
    representative = [by_name[name] for name in REPRESENTATIVE_DATASETS if name in by_name]
    remaining = [row for row in usable if str(row.get("dataset_name")).lower() not in set(REPRESENTATIVE_DATASETS)]
    if phase == "representative":
        return representative
    if phase == "remaining":
        return remaining
    return representative + remaining


def make_plan(inventory: list, selected_dataset: str, selected_config: str, selected_seed, root: Path, phase: str) -> list:
    datasets = phase_dataset_order(inventory, phase)
    if selected_dataset:
        datasets = [row for row in datasets if str(row.get("dataset_name")).lower() == selected_dataset.lower()]
    config_ids = [selected_config] if selected_config else list(CONFIGS)
    seeds = [int(selected_seed)] if selected_seed is not None else SEEDS
    plan = []
    for row in datasets:
        for cfg_id in config_ids:
            for seed in seeds:
                plan.append(make_run_config(str(row["dataset_name"]), cfg_id, seed, root))
    return plan


def command_for(cfg: dict, run_dir: Path, python_bin: str, device: str, q_chunk: int, row_block: int, batch_size: int) -> list:
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
        "batch_size": batch_size,
        "epoch": cfg["epoch"],
        "learning_rate": cfg["learning_rate"],
        "edge_dim": cfg["edge_dim"],
        "time_dim": cfg["time_dim"],
        "edge_hidden_dim": cfg["edge_hidden_dim"],
        "cluster_hidden_dim": cfg["cluster_hidden_dim"],
        "edge_encoder_mode": cfg["edge_encoder_mode"],
        "direct_time_scale": cfg["direct_time_scale"],
        "require_pretrained_node2vec": cfg["require_pretrained_node2vec"],
        "alpha": cfg["alpha"],
        "T": cfg["T"],
        "beta": cfg["beta"],
        "edge_neighbor_k": cfg["edge_neighbor_k"],
        "edge_ppr_topk": cfg["edge_ppr_topk"],
        "edge_ppr_method": cfg["edge_ppr_method"],
        "forest_samples": cfg["forest_samples"],
        "ncut_scope": "global",
        "cluster_loss_type": "matrix_ncut",
        "orth_type": cfg["orth_type"],
        "lambda_orth": cfg["lambda_orth"],
        "global_q_chunk_size": q_chunk,
        "global_ncut_row_block_size": row_block,
        "global_warmup_epochs": cfg["global_warmup_epochs"],
        "quiet": 1,
        "lambda_prox": cfg["lambda_prox"],
        "lambda_edge_ncut": cfg["lambda_edge_ncut"],
        "lambda_proj": cfg["lambda_proj"],
        "lambda_bal": cfg["lambda_bal"],
        "lambda_node_anchor": cfg["lambda_node_anchor"],
        "node_emb_mode": cfg["node_emb_mode"],
        "node_emb_lr": cfg["node_emb_lr"],
        "prox_similarity_mode": cfg["prox_similarity_mode"],
        "prox_role_ss_weight": cfg["prox_role_ss_weight"],
        "prox_role_dd_weight": cfg["prox_role_dd_weight"],
        "prox_role_ds_weight": cfg["prox_role_ds_weight"],
        "prox_role_sd_weight": cfg["prox_role_sd_weight"],
        "prox_role_time_weight": cfg["prox_role_time_weight"],
        "prox_temperature": cfg["prox_temperature"],
        "cluster_output_bias_mode": cfg["cluster_output_bias_mode"],
        "cluster_input_norm": cfg["cluster_input_norm"],
        "cluster_init_mode": cfg["cluster_init_mode"],
        "prototype_sample_size": cfg["prototype_sample_size"],
        "prototype_lloyd_iters": cfg["prototype_lloyd_iters"],
        "direct_kmeans_eval": 0,
        "init_only": 0,
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


def mark_success(run_dir: Path, cfg: dict, q_chunk: int, row_block: int, batch_size: int) -> None:
    result_path = run_dir / "result.json"
    result = read_json(result_path)
    result.update(
        {
            "status": "success",
            "config": cfg.get("config"),
            "config_name": cfg.get("name"),
            "config_checksum": cfg.get("config_checksum"),
            "final_global_q_chunk_size": int(q_chunk),
            "final_global_ncut_row_block_size": int(row_block),
            "final_batch_size": int(batch_size),
            "run_config": cfg,
        }
    )
    write_json(result_path, result)


def run_one(root: Path, cfg: dict, out_dir: Path, python_bin: str, device: str, resume: bool) -> tuple:
    run_dir = run_dir_for(out_dir, cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and is_success_result(run_dir, cfg["config_checksum"]):
        return "skipped", run_dir
    q_values = [8192, 4096, 2048, 1024, 512]
    q_values = [q for q in q_values if q <= int(cfg["global_q_chunk_size"])] or [int(cfg["global_q_chunk_size"])]
    attempts = [(q, min(int(cfg["global_ncut_row_block_size"]), q * 8), int(cfg["batch_size"])) for q in q_values]
    attempts.append((q_values[-1], min(int(cfg["global_ncut_row_block_size"]), q_values[-1] * 8), max(1, int(cfg["batch_size"]) // 2)))
    attempts.append((q_values[-1], min(int(cfg["global_ncut_row_block_size"]), q_values[-1] * 8), max(1, int(cfg["batch_size"]) // 4)))
    started = time.time()
    for attempt_idx, (q_chunk, row_block, batch_size) in enumerate(attempts, start=1):
        cmd = command_for(cfg, run_dir, python_bin, device, q_chunk, row_block, batch_size)
        with (run_dir / "train.log").open("a", encoding="utf-8") as log:
            log.write(
                f"[START] dataset={cfg['dataset']} config={cfg['config']} direct_time_scale={cfg['direct_time_scale']} "
                f"node_emb_mode={cfg['node_emb_mode']} node_emb_lr={cfg['node_emb_lr']} "
                f"lambda_anchor={cfg['lambda_node_anchor']} prox_similarity={cfg['prox_similarity_mode']} "
                f"lambda_prox={cfg['lambda_prox']} seed={cfg['model_seed']} forest_seed={cfg['forest_seed']} "
                f"node2vec_path={resolve_node2vec_path(root, cfg['dataset'])} attempt={attempt_idx} "
                f"q_chunk={q_chunk} row_block={row_block} batch_size={batch_size}\n"
            )
            proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"[DONE] exit_code={proc.returncode} runtime={time.time() - started:.6f}\n")
        if proc.returncode == 0 and read_json(run_dir / "result.json").get("status") == "success":
            mark_success(run_dir, cfg, q_chunk, row_block, batch_size)
            return "success", run_dir
        tail = (run_dir / "train.log").read_text(encoding="utf-8", errors="ignore")[-5000:]
        if "out of memory" not in tail.lower() and "cuda oom" not in tail.lower():
            write_failure(run_dir, cfg, "runtime_error", tail, proc.returncode, time.time() - started)
            return "failed", run_dir
    tail = (run_dir / "train.log").read_text(encoding="utf-8", errors="ignore")[-5000:]
    write_failure(run_dir, cfg, "oom", tail, 1, time.time() - started)
    return "failed", run_dir


def prewarm_cache(root: Path, out_dir: Path, inventory: list, python_bin: str, device: str) -> dict:
    cache_info = {}
    for row in inventory:
        if not bool(row.get("usable")):
            continue
        dataset = str(row["dataset_name"])
        cfg = make_run_config(dataset, "N0", 42, root)
        cfg["epoch"] = 0
        run_dir = out_dir / "code_info" / "cache_prewarm" / dataset
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = command_for(cfg, run_dir, python_bin, device, int(cfg["global_q_chunk_size"]), int(cfg["global_ncut_row_block_size"]), int(cfg["batch_size"]))
        cmd.extend(["--init_only", "1"])
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        cache_info[dataset] = {"exit_code": proc.returncode, "run_dir": str(run_dir)}
    write_json(out_dir / "code_info" / "cache_info.json", cache_info)
    return cache_info


def latest_metric(run_dir: Path) -> dict:
    rows = read_csv_rows(run_dir / "metrics.csv")
    return rows[-1] if rows else {}


def diagnostic_stage(run_dir: Path, stage_name: str) -> dict:
    rows = read_csv_rows(run_dir / "diagnostic_summary.csv")
    for row in rows:
        if row.get("stage") == stage_name:
            return row
    return {}


def previous_reference(root: Path) -> dict:
    candidates = [
        root / "logs" / "direct_trainable_node_time_all_datasets" / "20260727_005307",
        root / "results" / "server_runs" / "direct_trainable_node_time_all_datasets" / "20260727_005307",
    ]
    for base in candidates:
        per = base / "per_dataset_summary.csv"
        cross = base / "cross_dataset_summary.csv"
        if per.exists():
            rows = read_csv_rows(per)
            e0 = {
                row.get("dataset", ""): {
                    "Final Macro-F1 mean": row.get("Final Macro-F1 mean", ""),
                    "Best Macro-F1 mean": row.get("Best Macro-F1 mean", ""),
                }
                for row in rows
                if row.get("config") == "E0"
            }
            return {
                "historical_reference_available": True,
                "path": str(base),
                "historical_E0_by_dataset": e0,
                "cross_dataset_summary": read_csv_rows(cross) if cross.exists() else [],
            }
    return {"historical_reference_available": False, "reason": "previous direct_node_time result directory not found"}


def collect_master(out_dir: Path, inventory: list) -> list:
    rows = []
    for result_path in out_dir.rglob("result.json"):
        if "/code_info/cache_prewarm/" in str(result_path).replace("\\", "/"):
            continue
        run_dir = result_path.parent
        result = read_json(result_path)
        cfg_json = read_json(run_dir / "config.json")
        metric = latest_metric(run_dir)
        final_stage = diagnostic_stage(run_dir, "final_epoch")
        initial_stage = diagnostic_stage(run_dir, "after_cluster_initialization")
        changes = result.get("training_change_metrics") or {}
        cfg = result.get("run_config") or {}
        row = {
            "run_dir": str(run_dir),
            "status": result.get("status", "missing"),
            "dataset": result.get("dataset", cfg_json.get("dataset", cfg.get("dataset", ""))),
            "config": result.get("config", cfg.get("config", "")),
            "config_name": result.get("config_name", cfg.get("name", "")),
            "direct_time_scale": result.get("direct_time_scale", cfg_json.get("direct_time_scale", cfg.get("direct_time_scale", ""))),
            "node_emb_mode": result.get("node_emb_mode", cfg_json.get("node_emb_mode", cfg.get("node_emb_mode", ""))),
            "node_emb_lr": cfg_json.get("node_emb_lr", cfg.get("node_emb_lr", "")),
            "lambda_node_anchor": result.get("lambda_node_anchor", cfg_json.get("lambda_node_anchor", cfg.get("lambda_node_anchor", ""))),
            "prox_similarity_mode": result.get("prox_similarity_mode", cfg_json.get("prox_similarity_mode", cfg.get("prox_similarity_mode", ""))),
            "lambda_prox": cfg_json.get("lambda_prox", cfg.get("lambda_prox", "")),
            "model_seed": result.get("model_seed", cfg_json.get("model_seed", cfg_json.get("seed", ""))),
            "forest_seed": result.get("forest_seed", cfg_json.get("forest_seed", "")),
            "best_epoch": result.get("best_epoch", ""),
            "MacroF1_init": changes.get("MacroF1_init", initial_stage.get("Macro_F1", "")),
            "MacroF1_best": (result.get("best_metrics") or {}).get("Macro_F1", ""),
            "MacroF1_final": (result.get("final_metrics") or {}).get("Macro_F1", metric.get("Macro_F1", "")),
            "NMI_final": (result.get("final_metrics") or {}).get("NMI", metric.get("NMI", "")),
            "ARI_final": (result.get("final_metrics") or {}).get("ARI", metric.get("ARI", "")),
            "ACC_final": (result.get("final_metrics") or {}).get("ACC", metric.get("ACC", "")),
            "runtime_seconds": result.get("runtime_seconds", metric.get("total_runtime_seconds", "")),
            "peak_gpu_memory_mb": metric.get("peak_gpu_memory_mb", ""),
            "node_embedding_relative_drift": changes.get("node_embedding_relative_drift", final_stage.get("node_embedding_relative_drift", "")),
            "node_embedding_cosine_to_initial_mean": changes.get("node_embedding_cosine_to_initial_mean", final_stage.get("node_embedding_cosine_to_initial_mean", "")),
            "node_update_from_prox": metric.get("node_update_from_prox", ""),
            "node_update_from_global": metric.get("node_update_from_global", ""),
            "node_update_prox_global_ratio": metric.get("node_update_prox_global_ratio", ""),
            "cluster_update_from_global": metric.get("cluster_update_from_global", ""),
            "event_degree_node_drift_spearman": changes.get("event_degree_node_drift_spearman", ""),
            "top_bottom_drift_ratio": changes.get("top_bottom_drift_ratio", ""),
            "source_block_norm_mean": final_stage.get("source_block_norm_mean", metric.get("source_block_norm_mean", "")),
            "destination_block_norm_mean": final_stage.get("destination_block_norm_mean", metric.get("destination_block_norm_mean", "")),
            "raw_time_block_norm_mean": final_stage.get("raw_time_block_norm_mean", metric.get("raw_time_block_norm_mean", "")),
            "scaled_time_block_norm_mean": final_stage.get("scaled_time_block_norm_mean", metric.get("scaled_time_block_norm_mean", "")),
            "scaled_time_to_node_ratio": final_stage.get("scaled_time_to_node_ratio", metric.get("scaled_time_to_node_ratio", "")),
            "active_node_clusters": final_stage.get("num_active_node_clusters", metric.get("node_hard_active_clusters", "")),
            "largest_node_ratio": final_stage.get("largest_node_cluster_ratio", metric.get("node_hard_largest_ratio", "")),
            "q_rank1_energy": final_stage.get("q_rank1_energy_ratio", metric.get("q_rank1_energy_ratio", "")),
            "q_centered_energy": final_stage.get("q_centered_energy", metric.get("q_centered_energy", "")),
            "macro_f1_delta_final_init": changes.get("macro_f1_delta_final_init", ""),
            "error_message": result.get("error_message", ""),
        }
        rows.append(row)
    for item in inventory:
        if bool(item.get("usable")):
            continue
        rows.append(
            {
                "run_dir": "",
                "status": "failed",
                "dataset": item.get("dataset_name", ""),
                "config": "inventory",
                "config_name": "unusable_dataset",
                "error_message": item.get("failure_reason", ""),
            }
        )
    return rows


def collapse(row: dict) -> bool:
    if finite(row.get("active_node_clusters"), 999) <= 1:
        return True
    if finite(row.get("largest_node_ratio"), 0.0) >= 0.95:
        return True
    return finite(row.get("q_rank1_energy"), 0.0) >= 0.9999 and finite(row.get("q_centered_energy"), 1e9) < 1.0


PER_DATASET_METRICS = [
    ("MacroF1_init", "Initial Macro-F1"),
    ("MacroF1_best", "Best Macro-F1"),
    ("MacroF1_final", "Final Macro-F1"),
    ("NMI_final", "Final NMI"),
    ("ARI_final", "Final ARI"),
    ("node_embedding_relative_drift", "Node Relative Drift"),
    ("node_embedding_cosine_to_initial_mean", "Node Cosine to Initial"),
    ("node_update_from_prox", "Node Update From Prox"),
    ("node_update_from_global", "Node Update From Global"),
    ("node_update_prox_global_ratio", "Prox/Global Update Ratio"),
    ("event_degree_node_drift_spearman", "Event Degree/Node Drift Spearman"),
    ("top_bottom_drift_ratio", "Top/Bottom Drift Ratio"),
    ("active_node_clusters", "Active Node Clusters"),
    ("largest_node_ratio", "Largest Node Ratio"),
    ("source_block_norm_mean", "Source Block Norm"),
    ("destination_block_norm_mean", "Destination Block Norm"),
    ("raw_time_block_norm_mean", "Raw Time Norm"),
    ("scaled_time_block_norm_mean", "Scaled Time Norm"),
    ("scaled_time_to_node_ratio", "Scaled Time/Node Ratio"),
    ("runtime_seconds", "Runtime"),
    ("peak_gpu_memory_mb", "Peak GPU Memory"),
]


def per_dataset_summary(master: list) -> list:
    groups = {}
    for row in master:
        if row.get("config") == "inventory":
            continue
        groups.setdefault((row.get("dataset", ""), row.get("config", "")), []).append(row)
    result = []
    for (dataset, config), rows in sorted(groups.items()):
        success = [r for r in rows if r.get("status") == "success"]
        out = {
            "dataset": dataset,
            "config": config,
            "config_name": CONFIGS.get(config, {}).get("name", ""),
            "Success Count": len(success),
            "Failure Count": len(rows) - len(success),
            "Collapse Count": sum(1 for r in success if collapse(r)),
        }
        if success:
            out.update(
                {
                    "direct_time_scale": success[0].get("direct_time_scale", ""),
                    "node_emb_mode": success[0].get("node_emb_mode", ""),
                    "node_emb_lr": success[0].get("node_emb_lr", ""),
                    "lambda_node_anchor": success[0].get("lambda_node_anchor", ""),
                    "prox_similarity_mode": success[0].get("prox_similarity_mode", ""),
                    "lambda_prox": success[0].get("lambda_prox", ""),
                }
            )
        for key, label in PER_DATASET_METRICS:
            mean, std = mean_std([r.get(key) for r in success])
            out[f"{label} mean"] = mean
            out[f"{label} std"] = std
        result.append(out)
    return result


def _paired_dataset_delta(by_dataset: dict, left: str, right: str) -> tuple:
    vals = []
    positive = negative = near_zero = 0
    for cfg_rows in by_dataset.values():
        if left in cfg_rows and right in cfg_rows:
            delta = finite(cfg_rows[left].get("Final Macro-F1 mean")) - finite(cfg_rows[right].get("Final Macro-F1 mean"))
            if math.isfinite(delta):
                vals.append(delta)
                if delta > 1e-6:
                    positive += 1
                elif delta < -1e-6:
                    negative += 1
                else:
                    near_zero += 1
    mean, std = mean_std(vals)
    return mean, std, positive, negative, near_zero


def cross_dataset_summary(per_rows: list, reference: dict) -> list:
    by_dataset = {}
    for row in per_rows:
        by_dataset.setdefault(row["dataset"], {})[row["config"]] = row
    rows = []
    wins = {cfg: 0 for cfg in CONFIGS}
    ranks = {cfg: [] for cfg in CONFIGS}
    for dataset, cfg_rows in by_dataset.items():
        scored = [(cfg, finite(row.get("Final Macro-F1 mean"))) for cfg, row in cfg_rows.items()]
        scored = [(cfg, score) for cfg, score in scored if math.isfinite(score)]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            wins[scored[0][0]] += 1
        for rank, (cfg, _) in enumerate(scored, start=1):
            ranks[cfg].append(rank)
    hist = reference.get("historical_E0_by_dataset", {}) if reference.get("historical_reference_available") else {}
    for cfg in CONFIGS:
        cfg_rows = [r for r in per_rows if r.get("config") == cfg]
        avg_final, std_final = mean_std([r.get("Final Macro-F1 mean") for r in cfg_rows])
        avg_best, std_best = mean_std([r.get("Best Macro-F1 mean") for r in cfg_rows])
        avg_rank, _ = mean_std(ranks.get(cfg, []))
        hist_delta = []
        for row in cfg_rows:
            h = hist.get(row.get("dataset", ""), {})
            if h:
                hist_delta.append(finite(row.get("Final Macro-F1 mean")) - finite(h.get("Final Macro-F1 mean")))
        hist_mean, hist_std = mean_std(hist_delta)
        rows.append(
            {
                "Config": cfg,
                "Average Final Macro-F1": avg_final,
                "Std Final Macro-F1": std_final,
                "Average Best Macro-F1": avg_best,
                "Std Best Macro-F1": std_best,
                "Average Dataset Rank": avg_rank,
                "Datasets Won": wins.get(cfg, 0),
                "Average Improvement Over Historical E0": hist_mean,
                "Std Improvement Over Historical E0": hist_std,
                "Collapse Total": sum(int(finite(r.get("Collapse Count"), 0)) for r in cfg_rows),
                "Failure Total": sum(int(finite(r.get("Failure Count"), 0)) for r in cfg_rows),
            }
        )
    return rows


def component_ablation_summary(per_rows: list, reference: dict) -> list:
    by_dataset = {}
    for row in per_rows:
        by_dataset.setdefault(row["dataset"], {})[row["config"]] = row
    comparisons = [
        ("time_scale_0_5", "N1", "N0"),
        ("time_scale_0_25", "N2", "N0"),
        ("time_scale_0_1", "N3", "N0"),
        ("small_lr_0_1", "N4", "N2"),
        ("small_lr_0_01", "N5", "N2"),
        ("weak_anchor", "N6", "N4"),
        ("stronger_anchor", "N7", "N4"),
        ("role_aware", "N8", "N7"),
        ("lambda_prox_0_1", "N9", "N8"),
        ("lambda_prox_0_01", "N10", "N8"),
    ]
    rows = []
    for name, left, right in comparisons:
        mean, std, pos, neg, zero = _paired_dataset_delta(by_dataset, left, right)
        rows.append(
            {
                "comparison": name,
                "left_config": left,
                "right_config": right,
                "Final Macro-F1 delta mean": mean,
                "Final Macro-F1 delta std": std,
                "positive_dataset_count": pos,
                "negative_dataset_count": neg,
                "near_zero_dataset_count": zero,
            }
        )
    best_rows = []
    hist = reference.get("historical_E0_by_dataset", {}) if reference.get("historical_reference_available") else {}
    for dataset, cfg_rows in by_dataset.items():
        n0 = finite(cfg_rows.get("N0", {}).get("Final Macro-F1 mean"))
        best_cfg = None
        best_score = -math.inf
        for cfg in [f"N{i}" for i in range(1, 11)]:
            score = finite(cfg_rows.get(cfg, {}).get("Final Macro-F1 mean"))
            if math.isfinite(score) and score > best_score:
                best_cfg, best_score = cfg, score
        hist_e0 = finite(hist.get(dataset, {}).get("Final Macro-F1 mean"))
        best_rows.append(
            {
                "comparison": "best_new_by_dataset",
                "dataset": dataset,
                "best_config": best_cfg,
                "best_new_final_macro_f1": best_score,
                "delta_best_new_vs_N0": best_score - n0 if math.isfinite(n0) and math.isfinite(best_score) else math.nan,
                "delta_best_new_vs_historical_E0": best_score - hist_e0 if math.isfinite(hist_e0) and math.isfinite(best_score) else math.nan,
            }
        )
    rows.extend(best_rows)
    return rows


def update_balance_summary(per_rows: list) -> list:
    rows = []
    for row in per_rows:
        rows.append(
            {
                "dataset": row.get("dataset", ""),
                "config": row.get("config", ""),
                "Node Relative Drift mean": row.get("Node Relative Drift mean", ""),
                "Node Cosine to Initial mean": row.get("Node Cosine to Initial mean", ""),
                "Node Update From Prox mean": row.get("Node Update From Prox mean", ""),
                "Node Update From Global mean": row.get("Node Update From Global mean", ""),
                "Prox/Global Update Ratio mean": row.get("Prox/Global Update Ratio mean", ""),
                "Event Degree/Node Drift Spearman mean": row.get("Event Degree/Node Drift Spearman mean", ""),
                "Top/Bottom Drift Ratio mean": row.get("Top/Bottom Drift Ratio mean", ""),
                "Final Macro-F1 mean": row.get("Final Macro-F1 mean", ""),
            }
        )
    return rows


def write_analysis(out_dir: Path, inventory: list, per_rows: list, cross_rows: list, component_rows: list, failed: list, reference: dict) -> None:
    usable = [r["dataset_name"] for r in inventory if bool(r.get("usable"))]
    lines = [
        "# ETGC Direct Node-Time Stabilization Analysis",
        "",
        "This report is generated from completed result.json, metrics.csv, and diagnostic.json files.",
        "Each dataset has equal weight in cross-dataset aggregates.",
        "",
        "## Current Status",
        f"- usable datasets: {', '.join(usable) if usable else 'none'}",
        f"- failed runs: {len(failed)}",
        f"- historical reference available: {reference.get('historical_reference_available')}",
        "",
        "## Configurations",
        "- N0: direct baseline.",
        "- N1-N3: direct time scaling.",
        "- N4-N5: small node learning rates.",
        "- N6-N7: Node2Vec anchor.",
        "- N8-N10: role-aware proximity with lambda_prox 1, 0.1, 0.01.",
    ]
    if cross_rows:
        best = sorted(cross_rows, key=lambda r: finite(r.get("Average Final Macro-F1"), -1.0), reverse=True)[0]
        lines.extend(["", "## Current Best By Final Mean", f"- config: {best.get('Config')}", f"- Average Final Macro-F1: {best.get('Average Final Macro-F1')}"])
    if component_rows:
        lines.append("")
        lines.append("## Main Ablations")
        for row in component_rows:
            if row.get("dataset"):
                continue
            lines.append(
                f"- {row.get('comparison')}: delta_mean={row.get('Final Macro-F1 delta mean')} "
                f"positive={row.get('positive_dataset_count')} negative={row.get('negative_dataset_count')}"
            )
    lines.extend(
        [
            "",
            "Use per_dataset_summary.csv, component_ablation_summary.csv, and update_balance_summary.csv for the required detailed answers.",
        ]
    )
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(out_dir: Path, inventory: list) -> dict:
    reference = read_json(out_dir / "code_info" / "previous_reference.json") or previous_reference(ROOT)
    master = collect_master(out_dir, inventory)
    write_csv(out_dir / "master_runs.csv", master)
    write_csv(out_dir / "combined_summary.csv", master)
    failed = [row for row in master if row.get("status") != "success"]
    write_csv(out_dir / "failed_runs.csv", failed)
    per_rows = per_dataset_summary(master)
    write_csv(out_dir / "per_dataset_summary.csv", per_rows)
    cross_rows = cross_dataset_summary(per_rows, reference)
    write_csv(out_dir / "cross_dataset_summary.csv", cross_rows)
    component_rows = component_ablation_summary(per_rows, reference)
    write_csv(out_dir / "component_ablation_summary.csv", component_rows)
    update_rows = update_balance_summary(per_rows)
    write_csv(out_dir / "update_balance_summary.csv", update_rows)
    write_analysis(out_dir, inventory, per_rows, cross_rows, component_rows, failed, reference)
    return {"master": len(master), "failed": len(failed), "usable_datasets": sum(1 for r in inventory if bool(r.get("usable")))}


def run_queue(args) -> dict:
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_datasets(root, COMMON_CONFIG)
    write_json(out_dir / "code_info" / "dataset_inventory.json", inventory)
    write_json(out_dir / "dataset_inventory.json", inventory)
    reference = previous_reference(root)
    write_json(out_dir / "code_info" / "previous_reference.json", reference)
    if args.prewarm:
        prewarm_cache(root, out_dir, inventory, args.python, args.device)
    plan = make_plan(inventory, args.dataset, args.config, args.seed, root, args.phase)
    for cfg in plan:
        status, run_dir = run_one(root, cfg, out_dir, args.python, args.device, args.resume)
        print(
            f"[QUEUE] dataset={cfg['dataset']} config={cfg['config']} seed={cfg['model_seed']} "
            f"status={status} run_dir={run_dir}",
            flush=True,
        )
        summarize(out_dir, inventory)
    return summarize(out_dir, inventory)


def parse_args():
    parser = argparse.ArgumentParser(description="Run and summarize ETGC direct node-time stabilization experiments.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--config", choices=list(CONFIGS), default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--phase", choices=["all", "representative", "remaining"], default="all")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    inventory_path = out_dir / "code_info" / "dataset_inventory.json"
    if args.run:
        summary = run_queue(args)
    else:
        inventory = read_json(inventory_path) or discover_datasets(root, COMMON_CONFIG)
        summary = summarize(out_dir, inventory)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
