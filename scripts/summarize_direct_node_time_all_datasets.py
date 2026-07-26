#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_data import load_edge_event_data
from edge_time import build_edge_time_features
from utils import hash_cfg


SEEDS = [42, 43, 44]
FOREST_SEED = 20260725
CONFIGS = {
    "E0": {
        "name": "E0_mlp_baseline",
        "edge_encoder_mode": "mlp",
        "node_emb_mode": "frozen",
        "lambda_prox": 1.0,
        "orth_type": "orth",
        "lambda_orth": 1.0,
    },
    "E1": {
        "name": "E1_direct_trainable",
        "edge_encoder_mode": "direct_node_time",
        "node_emb_mode": "full",
        "lambda_prox": 1.0,
        "orth_type": "orth",
        "lambda_orth": 1.0,
    },
    "E2": {
        "name": "E2_direct_frozen",
        "edge_encoder_mode": "direct_node_time",
        "node_emb_mode": "frozen",
        "lambda_prox": 1.0,
        "orth_type": "orth",
        "lambda_orth": 1.0,
    },
    "E3": {
        "name": "E3_direct_no_prox",
        "edge_encoder_mode": "direct_node_time",
        "node_emb_mode": "full",
        "lambda_prox": 0.0,
        "orth_type": "orth",
        "lambda_orth": 1.0,
    },
    "E4": {
        "name": "E4_direct_orthqa",
        "edge_encoder_mode": "direct_node_time",
        "node_emb_mode": "full",
        "lambda_prox": 1.0,
        "orth_type": "orthqa",
        "lambda_orth": 1.0,
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
    "alpha": 0.2,
    "T": 4,
    "beta": 5.0,
    "edge_neighbor_k": -1,
    "edge_ppr_topk": -1,
    "edge_ppr_method": "temporal_state_forest",
    "forest_samples": 5,
    "global_q_chunk_size": 8192,
    "global_ncut_row_block_size": 65536,
    "global_warmup_epochs": 0,
    "lambda_edge_ncut": 0.5,
    "lambda_proj": 0.0,
    "lambda_bal": 0.0,
    "node_emb_lr": 1e-5,
    "cluster_output_bias_mode": "none",
    "cluster_input_norm": "layernorm",
    "cluster_init_mode": "random_orthogonal",
    "prototype_sample_size": 20000,
    "prototype_lloyd_iters": 0,
    "require_pretrained_node2vec": 1,
}


def resolve_node2vec_path(root: Path, dataset: str, feature_path: str = "") -> str:
    if feature_path:
        return str(Path(feature_path).resolve())
    candidates = [
        root / "pretrain" / f"{dataset}_feature.emb",
        root / "emb" / f"{dataset}_feature.emb",
        root / "emb" / dataset / f"{dataset}_feature.emb",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def inspect_node2vec_file(path: str, num_nodes: int) -> dict:
    payload = {
        "node2vec_path": path,
        "node2vec_exists": os.path.exists(path),
        "node2vec_shape": None,
        "node2vec_dtype": "",
        "node2vec_complete": False,
        "node2vec_missing_count": None,
    }
    if not payload["node2vec_exists"]:
        return payload
    rows = {}
    with open(path, "r", encoding="utf-8") as reader:
        first = reader.readline().lstrip("\ufeff").strip().split()
        has_header = len(first) == 2 and all(tok.lstrip("-").isdigit() for tok in first)
        if first and not has_header:
            arr = [float(x) for x in first]
            rows[int(arr[0])] = arr[1:]
        for line in reader:
            line = line.strip()
            if not line:
                continue
            arr = [float(x) for x in line.split()]
            if len(arr) > 1:
                rows[int(arr[0])] = arr[1:]
    dim = len(next(iter(rows.values()))) if rows else 0
    missing = [nid for nid in range(int(num_nodes)) if nid not in rows or len(rows[nid]) != dim]
    payload.update(
        {
            "node2vec_shape": [int(num_nodes), int(dim)] if not missing and dim > 0 else [len(rows), int(dim)],
            "node2vec_dtype": "float32",
            "node2vec_complete": len(missing) == 0 and dim > 0,
            "node2vec_missing_count": len(missing),
        }
    )
    return payload


def ppr_cache_paths(root: Path, dataset: str, data, common: dict) -> dict:
    method = str(common["edge_ppr_method"])
    forest_impl = {
        "temporal_state_forest": "state_expanded_temporal_subdivision_forest",
        "forest": "state_expanded_temporal_subdivision_forest_alias",
        "legacy_temporal_forest": "legacy_original_node_temporal_forest",
        "truncated": "temporal_edge_event_truncated_ppr",
    }[method]
    cfg = {
        "dataset": dataset,
        "method": method,
        "alpha": float(common["alpha"]),
        "T": int(common["T"]),
        "forest_samples": int(common["forest_samples"]),
        "edge_neighbor_k": int(common["edge_neighbor_k"]),
        "edge_ppr_topk": int(common["edge_ppr_topk"]),
        "beta": float(common["beta"]),
        "seed": int(FOREST_SEED),
        "num_events": int(data.num_events),
        "forest_impl": forest_impl,
    }
    cfg_hash = hash_cfg(cfg)
    ds_cache = root / "cache" / dataset
    pi_path = ds_cache / f"edge_ppr_{method}_{cfg_hash}.npz"
    w_path = ds_cache / f"edge_ncut_affinity_{method}_{cfg_hash}.npz"
    return {
        "proximity_cache_hash": cfg_hash,
        "proximity_cache_exists": pi_path.exists() and w_path.exists(),
        "pi_cache_path": str(pi_path),
        "w_cache_path": str(w_path),
    }


def discover_datasets(root: Path, common: dict) -> list:
    data_root = root / "dataset"
    rows = []
    if not data_root.exists():
        return [
            {
                "dataset_name": "",
                "dataset_path": str(data_root),
                "usable": False,
                "failure_reason": "dataset root does not exist",
            }
        ]
    for child in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        name = child.name
        row = {
            "dataset_name": name,
            "dataset_path": str(child),
            "node_count": "",
            "event_count": "",
            "timestamp_count": "",
            "class_count": "",
            "node2vec_path": resolve_node2vec_path(root, name),
            "node2vec_exists": False,
            "node2vec_shape": None,
            "node2vec_dtype": "",
            "time_feature_shape": None,
            "proximity_cache_exists": False,
            "usable": False,
            "failure_reason": "",
        }
        if not (child / f"{name}.txt").exists():
            row["failure_reason"] = f"missing edge file: {child / (name + '.txt')}"
            rows.append(row)
            continue
        if not (child / "node2label.txt").exists():
            row["failure_reason"] = f"missing label file: {child / 'node2label.txt'}"
            rows.append(row)
            continue
        try:
            data = load_edge_event_data(str(data_root), name)
            time_feat = build_edge_time_features(data.src, data.dst, data.times, int(common["time_dim"]))
            row.update(
                {
                    "node_count": int(data.num_nodes),
                    "event_count": int(data.num_events),
                    "timestamp_count": int(len(set(float(x) for x in data.times.tolist()))),
                    "class_count": int(data.K),
                    "time_feature_shape": [int(x) for x in time_feat.shape],
                }
            )
            row.update(inspect_node2vec_file(row["node2vec_path"], data.num_nodes))
            row.update(ppr_cache_paths(root, name, data, common))
            if not row["node2vec_exists"]:
                row["failure_reason"] = f"missing Node2Vec embedding file: {row['node2vec_path']}"
            elif not row["node2vec_complete"]:
                row["failure_reason"] = (
                    f"incomplete Node2Vec embedding file: missing_or_bad_count={row['node2vec_missing_count']}"
                )
            else:
                row["usable"] = True
        except Exception as exc:
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    rows.sort(key=lambda x: (not bool(x.get("usable")), int(x.get("event_count") or 10**18), str(x.get("dataset_name"))))
    return rows


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as reader:
        return list(csv.DictReader(reader))


def write_csv(path: Path, rows: list, fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow({key: row.get(key, "") for key in fieldnames})


def stable_config_checksum(cfg: dict) -> str:
    keys = [
        "dataset",
        "config",
        "edge_encoder_mode",
        "node_emb_mode",
        "lambda_prox",
        "orth_type",
        "lambda_orth",
        "model_seed",
        "forest_seed",
        "epoch",
        "learning_rate",
        "node_emb_lr",
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


def is_success_result(run_dir: Path, checksum: str = "") -> bool:
    result = read_json(run_dir / "result.json")
    if result.get("status") != "success":
        return False
    if checksum and result.get("config_checksum") != checksum:
        return False
    return True


def run_dir_for(out_dir: Path, cfg: dict) -> Path:
    return out_dir / str(cfg["dataset"]).lower() / cfg["name"] / f"seed_{cfg['model_seed']}"


def make_run_config(dataset: str, config_id: str, seed: int, root: Path, out_dir: Path) -> dict:
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


def make_plan(inventory: list, selected_dataset: str, selected_config: str, selected_seed, root: Path, out_dir: Path) -> list:
    usable = [row for row in inventory if bool(row.get("usable"))]
    if selected_dataset:
        usable = [row for row in usable if str(row.get("dataset_name")).lower() == selected_dataset.lower()]
    config_ids = [selected_config] if selected_config else ["E0", "E1", "E2", "E3", "E4"]
    seeds = [int(selected_seed)] if selected_seed is not None else SEEDS
    plan = []
    for row in sorted(usable, key=lambda x: int(x.get("event_count") or 10**18)):
        dataset = str(row["dataset_name"])
        for config_id in config_ids:
            for seed in seeds:
                plan.append(make_run_config(dataset, config_id, seed, root, out_dir))
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
        "require_pretrained_node2vec": cfg["require_pretrained_node2vec"],
        "alpha": cfg["alpha"],
        "T": cfg["T"],
        "beta": cfg["beta"],
        "edge_neighbor_k": cfg["edge_neighbor_k"],
        "edge_ppr_topk": cfg["edge_ppr_topk"],
        "edge_ppr_method": cfg["edge_ppr_method"],
        "forest_samples": cfg["forest_samples"],
        "ncut_scope": "global",
        "cluster_loss_type": "trace_mincut",
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
        "node_emb_mode": cfg["node_emb_mode"],
        "node_emb_lr": cfg["node_emb_lr"],
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


def write_failure(run_dir: Path, cfg: dict, error_type: str, message: str, exit_code: int, runtime: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(cfg)
    payload.update(
        {
            "status": "failed",
            "error_type": error_type,
            "error_message": str(message)[:1000],
            "traceback_tail": str(message)[-3000:],
            "exit_code": int(exit_code),
            "runtime_seconds": float(runtime),
            "config_checksum": cfg.get("config_checksum", ""),
        }
    )
    write_json(run_dir / "result.json", payload)


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
    attempts = []
    for q_chunk in q_values:
        attempts.append((q_chunk, min(int(cfg["global_ncut_row_block_size"]), q_chunk * 8), int(cfg["batch_size"])))
    attempts.append((q_values[-1], min(int(cfg["global_ncut_row_block_size"]), q_values[-1] * 8), max(1, int(cfg["batch_size"]) // 2)))
    attempts.append((q_values[-1], min(int(cfg["global_ncut_row_block_size"]), q_values[-1] * 8), max(1, int(cfg["batch_size"]) // 4)))

    started = time.time()
    for attempt_idx, (q_chunk, row_block, batch_size) in enumerate(attempts[:5], start=1):
        cmd = command_for(cfg, run_dir, python_bin, device, q_chunk, row_block, batch_size)
        with (run_dir / "train.log").open("a", encoding="utf-8") as log:
            log.write(
                f"[START] dataset={cfg['dataset']} config={cfg['config']} "
                f"edge_encoder_mode={cfg['edge_encoder_mode']} node_emb_mode={cfg['node_emb_mode']} "
                f"lambda_prox={cfg['lambda_prox']} orth_type={cfg['orth_type']} seed={cfg['model_seed']} "
                f"forest_seed={cfg['forest_seed']} node2vec_path={resolve_node2vec_path(root, cfg['dataset'])} "
                f"attempt={attempt_idx} q_chunk={q_chunk} row_block={row_block} batch_size={batch_size}\n"
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
        cfg = make_run_config(dataset, "E0", 42, root, out_dir)
        cfg["epoch"] = 0
        run_dir = out_dir / "code_info" / "cache_prewarm" / dataset
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = command_for(
            cfg,
            run_dir,
            python_bin,
            device,
            int(cfg["global_q_chunk_size"]),
            int(cfg["global_ncut_row_block_size"]),
            int(cfg["batch_size"]),
        )
        cmd.extend(["--init_only", "1"])
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        cache_info[dataset] = {"exit_code": proc.returncode, "run_dir": str(run_dir)}
    write_json(out_dir / "code_info" / "cache_info.json", cache_info)
    return cache_info


def finite(value, default=math.nan):
    try:
        if value in ("", None):
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def mean_std(values) -> tuple:
    vals = [finite(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return math.nan, math.nan
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, var ** 0.5


def latest_metric(run_dir: Path) -> dict:
    rows = read_csv_rows(run_dir / "metrics.csv")
    return rows[-1] if rows else {}


def diagnostic_stage(run_dir: Path, stage_name: str) -> dict:
    rows = read_csv_rows(run_dir / "diagnostic_summary.csv")
    for row in rows:
        if row.get("stage") == stage_name:
            return row
    return {}


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
            "edge_encoder_mode": result.get("edge_encoder_mode", cfg_json.get("edge_encoder_mode", "")),
            "node_emb_mode": result.get("node_emb_mode", cfg_json.get("node_emb_mode", "")),
            "lambda_prox": cfg_json.get("lambda_prox", cfg.get("lambda_prox", "")),
            "orth_type": cfg_json.get("orth_type", cfg.get("orth_type", "")),
            "lambda_orth": cfg_json.get("lambda_orth", cfg.get("lambda_orth", "")),
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
            "node_embedding_drift_fro_normalized": changes.get("node_embedding_drift_fro_normalized", final_stage.get("node_embedding_drift_fro_normalized", "")),
            "node_embedding_relative_drift": changes.get("node_embedding_relative_drift", final_stage.get("node_embedding_relative_drift", "")),
            "node_embedding_cosine_to_initial_mean": changes.get("node_embedding_cosine_to_initial_mean", final_stage.get("node_embedding_cosine_to_initial_mean", "")),
            "source_block_norm_mean": final_stage.get("source_block_norm_mean", metric.get("source_block_norm_mean", "")),
            "destination_block_norm_mean": final_stage.get("destination_block_norm_mean", metric.get("destination_block_norm_mean", "")),
            "time_block_norm_mean": final_stage.get("time_block_norm_mean", metric.get("time_block_norm_mean", "")),
            "edge_repr_common_to_variation_ratio": final_stage.get("feature_common_to_variation_ratio_before_norm", metric.get("feature_common_to_variation_ratio_before_norm", "")),
            "node_grad_l2_from_prox": final_stage.get("node_grad_l2_from_prox", metric.get("node_grad_l2_from_prox", "")),
            "node_grad_l2_from_cut": final_stage.get("node_grad_l2_from_cut", metric.get("node_grad_l2_from_cut", "")),
            "node_grad_l2_from_penalty": final_stage.get("node_grad_l2_from_penalty", metric.get("node_grad_l2_from_penalty", "")),
            "active_edge_clusters": final_stage.get("num_active_edge_clusters", metric.get("edge_hard_active_clusters", "")),
            "active_node_clusters": final_stage.get("num_active_node_clusters", metric.get("node_hard_active_clusters", "")),
            "largest_edge_ratio": final_stage.get("largest_edge_cluster_ratio", metric.get("edge_hard_largest_ratio", "")),
            "largest_node_ratio": final_stage.get("largest_node_cluster_ratio", metric.get("node_hard_largest_ratio", "")),
            "q_entropy": final_stage.get("q_entropy_mean", metric.get("Q_mean_entropy", "")),
            "q_rank1_energy": final_stage.get("q_rank1_energy_ratio", metric.get("q_rank1_energy_ratio", "")),
            "q_centered_energy": final_stage.get("q_centered_energy", metric.get("q_centered_energy", "")),
            "macro_f1_delta_final_init": changes.get("macro_f1_delta_final_init", ""),
            "macro_f1_delta_best_init": changes.get("macro_f1_delta_best_init", ""),
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


PER_DATASET_METRICS = [
    ("MacroF1_init", "Initial Macro-F1"),
    ("MacroF1_best", "Best Macro-F1"),
    ("MacroF1_final", "Final Macro-F1"),
    ("NMI_final", "Final NMI"),
    ("ARI_final", "Final ARI"),
    ("ACC_final", "Final ACC"),
    ("node_embedding_drift_fro_normalized", "Node Embedding Drift"),
    ("node_embedding_relative_drift", "Node Embedding Relative Drift"),
    ("node_embedding_cosine_to_initial_mean", "Node Embedding Cosine to Initial"),
    ("source_block_norm_mean", "Source Block Norm"),
    ("destination_block_norm_mean", "Destination Block Norm"),
    ("time_block_norm_mean", "Time Block Norm"),
    ("edge_repr_common_to_variation_ratio", "Event Representation Common/Variation Ratio"),
    ("node_grad_l2_from_prox", "Node Gradient From Prox"),
    ("node_grad_l2_from_cut", "Node Gradient From Cut"),
    ("node_grad_l2_from_penalty", "Node Gradient From Penalty"),
    ("active_edge_clusters", "Active Edge Clusters"),
    ("active_node_clusters", "Active Node Clusters"),
    ("largest_edge_ratio", "Largest Edge Ratio"),
    ("largest_node_ratio", "Largest Node Ratio"),
    ("q_entropy", "Q Entropy"),
    ("q_rank1_energy", "Q Rank1 Energy"),
    ("q_centered_energy", "Q Centered Energy"),
    ("runtime_seconds", "Runtime"),
    ("peak_gpu_memory_mb", "Peak GPU Memory"),
]


def collapse(row: dict) -> bool:
    if finite(row.get("active_node_clusters"), 999) <= 1:
        return True
    if finite(row.get("largest_node_ratio"), 0.0) >= 0.95:
        return True
    return finite(row.get("q_rank1_energy"), 0.0) >= 0.9999 and finite(row.get("q_centered_energy"), 1e9) < 1.0


def per_dataset_summary(master: list) -> list:
    groups = {}
    for row in master:
        if row.get("config") == "inventory":
            continue
        key = (row.get("dataset", ""), row.get("config", ""))
        groups.setdefault(key, []).append(row)
    result = []
    for (dataset, config), rows in sorted(groups.items()):
        success = [r for r in rows if r.get("status") == "success"]
        out = {
            "dataset": dataset,
            "config": config,
            "config_name": success[0].get("config_name", rows[0].get("config_name", "")) if rows else "",
            "Success Count": len(success),
            "Failure Count": len(rows) - len(success),
            "Collapse Count": sum(1 for r in success if collapse(r)),
        }
        for key, label in PER_DATASET_METRICS:
            mean, std = mean_std([r.get(key) for r in success])
            out[f"{label} mean"] = mean
            out[f"{label} std"] = std
        result.append(out)
    return result


def cross_dataset_summary(per_rows: list) -> list:
    by_dataset = {}
    for row in per_rows:
        by_dataset.setdefault(row["dataset"], {})[row["config"]] = row
    ranks = {cfg: [] for cfg in CONFIGS}
    wins = {cfg: 0 for cfg in CONFIGS}
    improved_e0 = {cfg: 0 for cfg in CONFIGS}
    deltas_e0 = {cfg: [] for cfg in CONFIGS}
    for dataset, cfg_rows in by_dataset.items():
        scored = []
        e0 = finite(cfg_rows.get("E0", {}).get("Final Macro-F1 mean"))
        for cfg_id, row in cfg_rows.items():
            score = finite(row.get("Final Macro-F1 mean"))
            if math.isfinite(score):
                scored.append((cfg_id, score))
                if cfg_id != "E0" and math.isfinite(e0):
                    delta = score - e0
                    deltas_e0[cfg_id].append(delta)
                    if delta > 0:
                        improved_e0[cfg_id] += 1
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            wins[scored[0][0]] += 1
        for rank, (cfg_id, _) in enumerate(scored, start=1):
            ranks[cfg_id].append(rank)

    rows = []
    for cfg_id in CONFIGS:
        cfg_rows = [r for r in per_rows if r.get("config") == cfg_id]
        avg_final, _ = mean_std([r.get("Final Macro-F1 mean") for r in cfg_rows])
        avg_best, _ = mean_std([r.get("Best Macro-F1 mean") for r in cfg_rows])
        avg_nmi, _ = mean_std([r.get("Final NMI mean") for r in cfg_rows])
        avg_ari, _ = mean_std([r.get("Final ARI mean") for r in cfg_rows])
        avg_rank, _ = mean_std(ranks.get(cfg_id, []))
        avg_delta_e0, _ = mean_std(deltas_e0.get(cfg_id, []))
        rows.append(
            {
                "Config": cfg_id,
                "Average Final Macro-F1": avg_final,
                "Average Best Macro-F1": avg_best,
                "Average Final NMI": avg_nmi,
                "Average Final ARI": avg_ari,
                "Average Dataset Rank": avg_rank,
                "Datasets Won": wins.get(cfg_id, 0),
                "Datasets Improved Over E0": improved_e0.get(cfg_id, 0),
                "Average Improvement Over E0": avg_delta_e0,
                "Average Improvement E1 Over E2": _paired_delta(by_dataset, "E1", "E2"),
                "Average Improvement E1 Over E3": _paired_delta(by_dataset, "E1", "E3"),
                "Average Improvement E4 Over E1": _paired_delta(by_dataset, "E4", "E1"),
                "Collapse Total": sum(int(finite(r.get("Collapse Count"), 0)) for r in cfg_rows),
                "Failure Total": sum(int(finite(r.get("Failure Count"), 0)) for r in cfg_rows),
            }
        )
    return rows


def _paired_delta(by_dataset: dict, left: str, right: str) -> float:
    vals = []
    for cfg_rows in by_dataset.values():
        if left in cfg_rows and right in cfg_rows:
            vals.append(finite(cfg_rows[left].get("Final Macro-F1 mean")) - finite(cfg_rows[right].get("Final Macro-F1 mean")))
    mean, _ = mean_std(vals)
    return mean


def write_analysis(out_dir: Path, inventory: list, per_rows: list, cross_rows: list, failed: list) -> None:
    usable = [r["dataset_name"] for r in inventory if bool(r.get("usable"))]
    unusable = [r for r in inventory if not bool(r.get("usable"))]
    lines = [
        "# ETGC Direct Node-Time Analysis",
        "",
        "This report is generated from completed result.json, metrics.csv, and diagnostic.json files.",
        "Each dataset has equal weight in cross-dataset aggregates.",
        "",
        "## Current Status",
        f"- usable datasets: {', '.join(usable) if usable else 'none'}",
        f"- unusable datasets: {len(unusable)}",
        f"- failed runs: {len(failed)}",
        "",
        "## Required Comparisons",
        "- E1 vs E0: direct trainable node-time representation relative to the MLP baseline.",
        "- E2 vs E0: direct frozen representation relative to the MLP baseline.",
        "- E1 vs E2: trainable Node2Vec relative to frozen Node2Vec.",
        "- E1 vs E3: proximity contribution in direct mode.",
        "- E4 vs E1: orthqa relative to orth in direct mode.",
        "",
        "Use per_dataset_summary.csv for dataset-level means/stds and cross_dataset_summary.csv for macro-averaged conclusions.",
    ]
    if cross_rows:
        best = sorted(cross_rows, key=lambda r: finite(r.get("Average Final Macro-F1"), -1.0), reverse=True)[0]
        lines.extend(["", "## Current Best By Final Mean", f"- config: {best.get('Config')}", f"- Average Final Macro-F1: {best.get('Average Final Macro-F1')}"])
    if unusable:
        lines.append("")
        lines.append("## Unusable Datasets")
        for row in unusable:
            lines.append(f"- {row.get('dataset_name')}: {row.get('failure_reason')}")
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(out_dir: Path, inventory: list) -> dict:
    master = collect_master(out_dir, inventory)
    write_csv(out_dir / "master_runs.csv", master)
    write_csv(out_dir / "combined_summary.csv", master)
    failed = [row for row in master if row.get("status") != "success"]
    write_csv(out_dir / "failed_runs.csv", failed)
    per_rows = per_dataset_summary(master)
    write_csv(out_dir / "per_dataset_summary.csv", per_rows)
    cross_rows = cross_dataset_summary(per_rows)
    write_csv(out_dir / "cross_dataset_summary.csv", cross_rows)
    write_analysis(out_dir, inventory, per_rows, cross_rows, failed)
    return {"master": len(master), "failed": len(failed), "usable_datasets": sum(1 for r in inventory if bool(r.get("usable")))}


def run_queue(args) -> dict:
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_datasets(root, COMMON_CONFIG)
    write_json(out_dir / "code_info" / "dataset_inventory.json", inventory)
    write_json(out_dir / "dataset_inventory.json", inventory)
    if args.prewarm:
        prewarm_cache(root, out_dir, inventory, args.python, args.device)
    plan = make_plan(inventory, args.dataset, args.config, args.seed, root, out_dir)
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
    parser = argparse.ArgumentParser(description="Run and summarize ETGC direct node-time experiments.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--config", choices=["E0", "E1", "E2", "E3", "E4"], default="")
    parser.add_argument("--seed", type=int, default=None)
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
