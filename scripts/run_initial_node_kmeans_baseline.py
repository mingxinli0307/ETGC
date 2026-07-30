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
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_data import load_edge_event_data  # noqa: E402
from edge_metrics import evaluate_node_clustering  # noqa: E402
from edge_model import (  # noqa: E402
    assign_all_to_centers,
    feature_common_variation_statistics,
    fit_kmeans_centers,
    load_pretrained_node_features,
    tensor_checksum,
)
from scripts.summarize_direct_node_time_all_datasets import (  # noqa: E402
    COMMON_CONFIG as INVENTORY_COMMON_CONFIG,
    discover_datasets,
    inspect_node2vec_file,
    resolve_node2vec_path,
    write_csv,
    write_json,
)


DEFAULT_SEEDS = [42, 43, 44]


def parse_int_list(value: str) -> list:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_name_list(value: str) -> list:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run_command(cmd: list, cwd: Path = ROOT) -> str:
    try:
        result = subprocess.run(cmd, cwd=str(cwd), check=False, text=True, capture_output=True)
        text = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if err:
            text = f"{text}\n{err}".strip()
        return text
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def stable_config_checksum(cfg: dict) -> str:
    keys = [
        "method",
        "dataset",
        "seed",
        "node_count",
        "class_count",
        "node2vec_path",
        "node_kmeans_sample_size",
        "node_kmeans_lloyd_iters",
        "assign_chunk_size",
        "device",
        "require_pretrained_node2vec",
    ]
    payload = {key: cfg.get(key) for key in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_success_result(run_dir: Path, checksum: str) -> bool:
    payload = read_json(run_dir / "result.json")
    return payload.get("status") == "success" and payload.get("config_checksum") == checksum


def hard_node_cluster_statistics(pred_y: np.ndarray, K: int) -> dict:
    pred = np.asarray(pred_y, dtype=np.int64)
    K = int(K)
    if pred.ndim != 1:
        raise ValueError(f"pred_y must be 1D, got shape={pred.shape}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    n = int(pred.shape[0])
    counts = np.bincount(pred, minlength=K)[:K].astype(np.int64)
    if int(counts.sum()) != n:
        raise ValueError(f"Node hard cluster counts do not sum to N: counts_sum={counts.sum()}, N={n}")
    ratios = counts.astype(np.float64) / max(n, 1)
    if n > 0 and abs(float(ratios.sum()) - 1.0) > 1e-6:
        raise ValueError(f"Node hard cluster ratios do not sum to 1: ratio_sum={ratios.sum()}")
    nonempty = ratios[counts > 0]
    return {
        "node_hard_counts": counts.tolist(),
        "node_hard_ratios": ratios.tolist(),
        "node_hard_active_clusters": int(np.sum(counts > 0)),
        "node_hard_empty_clusters": int(np.sum(counts == 0)),
        "node_hard_largest_ratio": float(ratios.max()) if ratios.size else 0.0,
        "node_hard_smallest_nonempty_ratio": float(nonempty.min()) if nonempty.size else 0.0,
    }


def node_feature_statistics(features: torch.Tensor) -> dict:
    stats = feature_common_variation_statistics(features)
    norm = torch.linalg.norm(features.detach(), dim=1)
    stats.update(
        {
            "node_embedding_norm_mean": float(norm.mean().cpu()) if norm.numel() else 0.0,
            "node_embedding_norm_std": float(norm.std(unbiased=False).cpu()) if norm.numel() else 0.0,
            "node_embedding_abs_max": float(features.detach().abs().max().cpu()) if features.numel() else 0.0,
        }
    )
    return stats


def compute_kmeans_inertia(
    features: torch.Tensor,
    centers: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 8192,
) -> float:
    total = 0.0
    chunk_size = max(1, int(chunk_size))
    for start in range(0, int(features.size(0)), chunk_size):
        end = min(start + chunk_size, int(features.size(0)))
        chunk = features[start:end]
        assigned = centers.index_select(0, labels[start:end].long())
        total += float((chunk - assigned).square().sum().detach().cpu())
    return total


def evaluate_initial_node_kmeans_features(
    features,
    labels,
    K: int,
    seed: int,
    sample_size: int = -1,
    lloyd_iters: int = 10,
    assign_chunk_size: int = 8192,
    device: str = "cpu",
) -> dict:
    """Run KMeans directly on initial node features and evaluate node labels.

    This baseline intentionally accepts no edge list and performs no incidence
    projection, so the prediction is exactly argmin over KMeans centers in H0.
    """
    labels_np = np.asarray(labels, dtype=np.int64)
    features_t = torch.as_tensor(features, dtype=torch.float32, device=torch.device(device))
    if features_t.dim() != 2:
        raise ValueError(f"features must be 2D, got shape={tuple(features_t.shape)}")
    if labels_np.shape[0] != int(features_t.size(0)):
        raise ValueError(f"labels length must equal node count: labels={labels_np.shape[0]}, N={features_t.size(0)}")
    effective_sample_size = int(features_t.size(0)) if int(sample_size) <= 0 else int(sample_size)
    with torch.no_grad():
        centers, fit_stats = fit_kmeans_centers(
            features_t,
            K=int(K),
            seed=int(seed),
            sample_size=effective_sample_size,
            max_iters=int(lloyd_iters),
        )
        pred_t = assign_all_to_centers(features_t, centers, chunk_size=int(assign_chunk_size))
        inertia = compute_kmeans_inertia(features_t, centers, pred_t, chunk_size=int(assign_chunk_size))
    pred_y = pred_t.detach().cpu().numpy().astype(np.int64)
    metrics = evaluate_node_clustering(labels_np, pred_y)
    stats = hard_node_cluster_statistics(pred_y, int(K))
    feature_stats = node_feature_statistics(features_t)
    result = {
        **metrics,
        **stats,
        **feature_stats,
        **fit_stats,
        "method": "initial_node_kmeans",
        "representation_source": "initial_node_features",
        "uses_edges": False,
        "uses_time": False,
        "uses_incidence_projection": False,
        "uses_training": False,
        "node_count": int(features_t.size(0)),
        "feature_dim": int(features_t.size(1)),
        "class_count": int(K),
        "seed": int(seed),
        "node_kmeans_sample_size": int(sample_size),
        "node_kmeans_sample_size_effective": int(effective_sample_size),
        "node_kmeans_lloyd_iters": int(lloyd_iters),
        "kmeans_inertia": float(inertia),
        "kmeans_inertia_per_node": float(inertia / max(int(features_t.size(0)), 1)),
        "node_feature_checksum": tensor_checksum(features_t.detach().cpu()),
    }
    return result


def load_initial_node_features_for_kmeans(
    node2vec_path: str,
    num_nodes: int,
    fallback_dim: int,
    seed: int,
    require_pretrained_node2vec: bool,
) -> np.ndarray:
    return load_pretrained_node_features(
        node2vec_path,
        int(num_nodes),
        int(fallback_dim),
        int(seed),
        require_existing=bool(require_pretrained_node2vec),
    )


def run_dir_for(out_dir: Path, dataset: str, seed: int) -> Path:
    return out_dir / str(dataset) / f"seed_{int(seed)}"


def write_failure(run_dir: Path, cfg: dict, exc: BaseException, runtime_seconds: float) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "config": cfg,
        "config_checksum": stable_config_checksum(cfg),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-20:]),
        "runtime_seconds": float(runtime_seconds),
    }
    write_json(run_dir / "result.json", payload)
    write_json(run_dir / "config.json", cfg)
    (run_dir / "train.log").write_text(
        f"[FAILED] dataset={cfg.get('dataset')} seed={cfg.get('seed')} "
        f"error_type={type(exc).__name__} error_message={str(exc)}\n",
        encoding="utf-8",
    )
    return payload


def run_one(root: Path, out_dir: Path, cfg: dict, resume: bool = False) -> dict:
    run_dir = run_dir_for(out_dir, cfg["dataset"], cfg["seed"])
    checksum = stable_config_checksum(cfg)
    if resume and is_success_result(run_dir, checksum):
        return read_json(run_dir / "result.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        data = load_edge_event_data(str(root / "dataset"), cfg["dataset"])
        features = load_initial_node_features_for_kmeans(
            cfg["node2vec_path"],
            data.num_nodes,
            cfg["fallback_dim"],
            cfg["seed"],
            bool(cfg["require_pretrained_node2vec"]),
        )
        metrics = evaluate_initial_node_kmeans_features(
            features,
            data.labels,
            data.K,
            seed=cfg["seed"],
            sample_size=cfg["node_kmeans_sample_size"],
            lloyd_iters=cfg["node_kmeans_lloyd_iters"],
            assign_chunk_size=cfg["assign_chunk_size"],
            device=cfg["device"],
        )
        runtime_seconds = time.time() - start
        row = {
            "dataset": cfg["dataset"],
            "seed": int(cfg["seed"]),
            "stage": "initial_node_kmeans",
            "status": "success",
            "runtime_seconds": float(runtime_seconds),
            **metrics,
        }
        csv_fields = [
            "dataset",
            "seed",
            "stage",
            "status",
            "ACC",
            "NMI",
            "ARI",
            "Macro_F1",
            "F1",
            "node_hard_active_clusters",
            "node_hard_largest_ratio",
            "node_hard_empty_clusters",
            "kmeans_inertia",
            "kmeans_inertia_per_node",
            "node_embedding_norm_mean",
            "node_embedding_norm_std",
            "feature_common_to_variation_ratio",
            "runtime_seconds",
        ]
        write_csv(run_dir / "metrics.csv", [row], fieldnames=csv_fields)
        diagnostic = {
            "config": cfg,
            "node_hard_counts": metrics["node_hard_counts"],
            "node_hard_ratios": metrics["node_hard_ratios"],
            "node_feature_stats": {
                key: value
                for key, value in metrics.items()
                if key.startswith("feature_") or key.startswith("node_embedding_")
            },
            "kmeans": {
                key: value for key, value in metrics.items() if key.startswith("kmeans_") or key == "kmeans_inertia"
            },
        }
        write_json(run_dir / "diagnostic.json", diagnostic)
        write_json(run_dir / "config.json", cfg)
        result = {
            "status": "success",
            "method": "initial_node_kmeans",
            "config": cfg,
            "config_checksum": checksum,
            "metrics": metrics,
            "best_metrics": metrics,
            "final_metrics": metrics,
            "best_epoch": 0,
            "final_epoch": 0,
            "runtime_seconds": float(runtime_seconds),
        }
        write_json(run_dir / "result.json", result)
        (run_dir / "train.log").write_text(
            "[initial_node_kmeans] "
            f"dataset={cfg['dataset']} seed={cfg['seed']} N={data.num_nodes} K={data.K} "
            f"node2vec_path={cfg['node2vec_path']} Macro_F1={metrics['Macro_F1']:.6f} "
            f"NMI={metrics['NMI']:.6f} ARI={metrics['ARI']:.6f} ACC={metrics['ACC']:.6f}\n",
            encoding="utf-8",
        )
        print(
            "[DONE] "
            f"dataset={cfg['dataset']} seed={cfg['seed']} "
            f"Macro_F1={metrics['Macro_F1']:.6f} NMI={metrics['NMI']:.6f} "
            f"ARI={metrics['ARI']:.6f} ACC={metrics['ACC']:.6f}"
        )
        return result
    except Exception as exc:
        payload = write_failure(run_dir, cfg, exc, time.time() - start)
        print(f"[FAILED] dataset={cfg['dataset']} seed={cfg['seed']} error={type(exc).__name__}: {exc}")
        return payload


def selected_inventory_rows(inventory: list, requested: list) -> list:
    if not requested:
        return inventory
    by_lower = {str(row.get("dataset_name", "")).lower(): row for row in inventory}
    rows = []
    for name in requested:
        row = by_lower.get(name.lower())
        if row is None:
            rows.append(
                {
                    "dataset_name": name,
                    "usable": False,
                    "failure_reason": f"requested dataset not found: {name}",
                }
            )
        else:
            rows.append(row)
    return rows


def make_run_configs(root: Path, inventory_rows: list, seeds: list, args) -> tuple:
    configs = []
    failures = []
    for row in inventory_rows:
        name = row.get("dataset_name", "")
        if not row.get("usable"):
            failures.append(
                {
                    "dataset": name,
                    "seed": "",
                    "status": "failed",
                    "error_type": "unusable_dataset",
                    "error_message": row.get("failure_reason", "unusable dataset"),
                }
            )
            continue
        for seed in seeds:
            cfg = {
                "method": "initial_node_kmeans",
                "dataset": name,
                "seed": int(seed),
                "node_count": int(row.get("node_count") or 0),
                "event_count": int(row.get("event_count") or 0),
                "class_count": int(row.get("class_count") or 0),
                "node2vec_path": resolve_node2vec_path(root, name, args.feature_path),
                "fallback_dim": int(args.fallback_dim),
                "node_kmeans_sample_size": int(args.node_kmeans_sample_size),
                "node_kmeans_lloyd_iters": int(args.node_kmeans_lloyd_iters),
                "assign_chunk_size": int(args.assign_chunk_size),
                "device": str(args.device),
                "require_pretrained_node2vec": int(args.require_pretrained_node2vec),
            }
            configs.append(cfg)
    return configs, failures


def flatten_result(result: dict) -> dict:
    cfg = result.get("config", {})
    metrics = result.get("metrics") or result.get("final_metrics") or {}
    row = {
        "dataset": cfg.get("dataset", ""),
        "seed": cfg.get("seed", ""),
        "status": result.get("status", ""),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
        "runtime_seconds": result.get("runtime_seconds", ""),
    }
    for key, value in metrics.items():
        if isinstance(value, (list, dict)):
            continue
        row[key] = value
    return row


def mean_std(values: list) -> tuple:
    finite = [float(v) for v in values if v is not None and v != "" and math.isfinite(float(v))]
    if not finite:
        return "", ""
    arr = np.asarray(finite, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize(out_dir: Path, extra_failures: list = None) -> None:
    results = []
    for path in sorted(out_dir.glob("*/*/result.json")):
        results.append(read_json(path))
    master = [flatten_result(result) for result in results]
    if extra_failures:
        master.extend(extra_failures)
    write_csv(out_dir / "master_runs.csv", master)
    failed = [row for row in master if row.get("status") != "success"]
    write_csv(out_dir / "failed_runs.csv", failed)
    grouped = {}
    for row in master:
        if row.get("status") == "success":
            grouped.setdefault(row.get("dataset", ""), []).append(row)
    summary_rows = []
    for dataset, rows in sorted(grouped.items(), key=lambda item: item[0].lower()):
        summary = {
            "dataset": dataset,
            "success_count": len(rows),
            "failure_count": len([r for r in master if r.get("dataset") == dataset and r.get("status") != "success"]),
        }
        for key in [
            "ACC",
            "NMI",
            "ARI",
            "Macro_F1",
            "node_hard_active_clusters",
            "node_hard_largest_ratio",
            "node_hard_empty_clusters",
            "kmeans_inertia_per_node",
            "feature_common_to_variation_ratio",
            "runtime_seconds",
        ]:
            mean, std = mean_std([row.get(key) for row in rows])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        summary_rows.append(summary)
    write_csv(out_dir / "summary.csv", summary_rows)
    analysis = [
        "# Initial Node KMeans Baseline",
        "",
        "This baseline runs KMeans directly on the initial Node2Vec node representation matrix H0.",
        "It does not use temporal edges, time features, incidence projection, ETGC training, or edge-event assignments.",
        "",
        f"Successful runs: {len([r for r in master if r.get('status') == 'success'])}",
        f"Failed runs: {len(failed)}",
        "",
        "## Per-Dataset Summary",
        "",
        "| Dataset | Final Macro-F1 Mean | Final Macro-F1 Std | NMI Mean | ARI Mean | ACC Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        analysis.append(
            f"| {row['dataset']} | {row.get('Macro_F1_mean', '')} | {row.get('Macro_F1_std', '')} | "
            f"{row.get('NMI_mean', '')} | {row.get('ARI_mean', '')} | {row.get('ACC_mean', '')} |"
        )
    if failed:
        analysis.extend(["", "## Failed Runs", ""])
        for row in failed:
            analysis.append(f"- {row.get('dataset')} seed={row.get('seed')}: {row.get('error_message')}")
    (out_dir / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")


def write_code_info(root: Path, out_dir: Path, inventory: list, args) -> None:
    code_dir = out_dir / "code_info"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "commit.txt").write_text(run_command(["git", "rev-parse", "HEAD"], root) + "\n", encoding="utf-8")
    env_lines = [
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(os.linesep, ' ')}",
        f"torch_version={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        env_lines.append(f"cuda_device={torch.cuda.get_device_name(0)}")
    (code_dir / "environment.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    write_json(code_dir / "dataset_inventory.json", inventory)
    write_json(out_dir / "dataset_inventory.json", inventory)
    write_json(
        code_dir / "common_config.json",
        {
            "method": "initial_node_kmeans",
            "seeds": parse_int_list(args.seeds),
            "node_kmeans_sample_size": int(args.node_kmeans_sample_size),
            "node_kmeans_lloyd_iters": int(args.node_kmeans_lloyd_iters),
            "assign_chunk_size": int(args.assign_chunk_size),
            "require_pretrained_node2vec": int(args.require_pretrained_node2vec),
            "device": str(args.device),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KMeans directly on initial Node2Vec node features.")
    parser.add_argument("--root", type=str, default=str(ROOT))
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated dataset names. Empty means all.")
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--feature_path", type=str, default="", help="Optional explicit Node2Vec file for one dataset.")
    parser.add_argument("--fallback_dim", type=int, default=128)
    parser.add_argument("--node_kmeans_sample_size", type=int, default=-1, help="<=0 means fit on all nodes.")
    parser.add_argument("--node_kmeans_lloyd_iters", type=int, default=10)
    parser.add_argument("--assign_chunk_size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--require_pretrained_node2vec", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all usable datasets. This is the default if datasets is empty.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    timestamp = args.timestamp or now_timestamp()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "logs" / "initial_node_kmeans" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_datasets(root, INVENTORY_COMMON_CONFIG)
    requested = parse_name_list(args.datasets)
    inventory_rows = selected_inventory_rows(inventory, requested)
    seeds = parse_int_list(args.seeds) or DEFAULT_SEEDS
    write_code_info(root, out_dir, inventory, args)
    configs, inventory_failures = make_run_configs(root, inventory_rows, seeds, args)
    inventory_failures = list(inventory_failures)
    for cfg in configs:
        print(
            "[START] "
            f"dataset={cfg['dataset']} seed={cfg['seed']} "
            f"node2vec_path={cfg['node2vec_path']} lloyd_iters={cfg['node_kmeans_lloyd_iters']}"
        )
        run_one(root, out_dir, cfg, resume=bool(args.resume))
        summarize(out_dir, inventory_failures)
    summarize(out_dir, inventory_failures)
    print(f"[DONE] output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
