#!/usr/bin/env python3
"""Run the fixed-seed ETGC lambda_esg x lambda_prox search with orth disabled."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


DATASETS = ("school", "dblp", "patent", "arXivAI")
EPOCHS = {"school": 100, "dblp": 120, "patent": 100, "arXivAI": 100}
VALUES = (0.0, 0.1, 0.25, 0.5, 1.0)
DATASET_COST = {"school": 1.0, "dblp": 5.0, "patent": 1.5, "arXivAI": 8.0}
FIXED = {
    "lambda_edge_ncut": 0.5,
    "lambda_orth": 0.0,
    "global_cut_scale": 1.0,
    "global_orth_scale": 0.0,
    "learning_rate": 1e-4,
    "node_emb_lr": 1e-5,
    "prototype_temperature": 0.2,
    "beta": 5.0,
    "alpha": 0.2,
    "T": 4,
    "edge_ppr_topk": 20,
    "forest_samples": 50,
    "edge_neighbor_k": -1,
    "affinity_sparsify": "symmetric_union_knn",
    "cluster_loss_type": "matrix_ncut",
    "ncut_scope": "global",
    "edge_encoder_mode": "direct_node_time",
    "time_feature_mode": "current",
    "direct_time_scale": 1.0,
    "cluster_head_type": "cosine_prototype",
    "prox_similarity_mode": "cosine",
    "prox_warmup_epochs": 5,
    "global_warmup_epochs": 0,
    "seed": 42,
    "model_seed": 42,
    "prototype_seed": 42,
    "forest_seed": 42,
    "cluster_init_mode": "random",
    "prototype_init_mode": "random",
    "node_prior_mode": "none",
    "lambda_node_prior": 0.0,
    "lambda_node_sbm": 0.0,
    "lambda_node_anchor": 0.0,
    "lambda_bal": 0.0,
    "lambda_proj": 0.0,
    "direct_kmeans_eval": 0,
    "direct_node_prior_eval": 0,
    "save_embeddings": 0,
}
ALLOWED_RUN_FILES = {"command.txt", "config.json", "train.log", "metrics.csv", "result.json"}
SUMMARY_FIELDS = (
    "dataset", "lambda_esg", "lambda_prox", "epochs", "status", "error",
    "final_ACC", "final_NMI", "final_ARI", "final_Macro_F1",
    "best_ACC", "best_ACC_epoch", "best_NMI", "best_NMI_epoch",
    "best_ARI", "best_ARI_epoch", "best_Macro_F1", "best_Macro_F1_epoch",
    "final_cut", "final_esg", "final_prox", "final_gain_mean", "final_gain_min", "final_gain_max",
    "final_rank1", "final_effective_rank", "final_q_margin", "final_volume_cv", "final_qtdq_condition",
    "final_uniform_l2", "final_entropy_gap", "final_node_up_prox", "final_node_up_global",
    "final_node_update_ratio", "mean_epoch_seconds", "mean_prox_seconds", "prox_optimizer_steps",
    "peak_gpu_memory_mb", "runtime_seconds", "assigned_physical_gpu",
)


@dataclass(frozen=True)
class Task:
    dataset: str
    lambda_esg: float
    lambda_prox: float

    @property
    def epochs(self) -> int:
        return EPOCHS[self.dataset]

    @property
    def slug(self) -> str:
        def value_slug(value: float) -> str:
            return str(value).replace(".", "p")

        return f"esg_{value_slug(self.lambda_esg)}_prox_{value_slug(self.lambda_prox)}"

    @property
    def estimated_cost(self) -> float:
        return DATASET_COST[self.dataset] * (1.0 + 3.0 * self.lambda_prox)


def tasks() -> list[Task]:
    return [Task(dataset, esg, prox) for dataset in DATASETS for esg in VALUES for prox in VALUES]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except Exception:
        return []


def number(value, default=math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def task_dir(output_dir: Path, task: Task) -> Path:
    return output_dir / task.dataset / task.slug


def task_complete(run_dir: Path, task: Task) -> bool:
    result = read_json(run_dir / "result.json")
    config = read_json(run_dir / "config.json")
    rows = read_csv(run_dir / "metrics.csv")
    if result.get("status") != "success" or len(rows) < task.epochs:
        return False
    expected = {
        "dataset": task.dataset,
        "epoch": task.epochs,
        "lambda_esg": task.lambda_esg,
        "lambda_prox": task.lambda_prox,
        "lambda_orth": 0.0,
        "global_orth_scale": 0.0,
        "cluster_loss_type": "matrix_ncut",
        "cluster_init_mode": "random",
        "prototype_init_mode": "random",
    }
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            if not math.isclose(number(actual), value, rel_tol=0.0, abs_tol=1e-12):
                return False
        elif actual != value:
            return False
    return int(number(rows[-1].get("epoch"), -1)) == task.epochs


def cleanup_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    for path in run_dir.iterdir():
        if path.name in ALLOWED_RUN_FILES:
            continue
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)


def diagnostic_epochs(epochs: int) -> str:
    return ",".join(str(value) for value in (1, 5, 10, 20, 30, 50, 75, 100, 120) if value <= epochs)


def build_command(args, task: Task, run_dir: Path) -> list[str]:
    values = {
        "dataset": task.dataset,
        "directed": 0,
        "device": "cuda:0",
        **FIXED,
        "data_root": str(args.asset_root / "dataset"),
        "emb_root": str(args.asset_root / "emb"),
        "pretrain_emb_dir": str(args.asset_root / "pretrain"),
        "cache_dir": str(args.asset_root / "cache"),
        "batch_size": 512,
        "epoch": task.epochs,
        "edge_dim": 128,
        "time_dim": 32,
        "edge_hidden_dim": 128,
        "cluster_hidden_dim": 64,
        "edge_ppr_method": "temporal_state_forest",
        "orth_type": "orthqa",
        "global_q_chunk_size": 8192,
        "global_ncut_row_block_size": 65536,
        "quiet": 1,
        "lambda_esg": task.lambda_esg,
        "lambda_prox": task.lambda_prox,
        "node_emb_mode": "small_lr",
        "cluster_output_bias_mode": "default",
        "cluster_input_norm": "none",
        "prototype_sample_size": 20000,
        "prototype_lloyd_iters": 10,
        "overnight_diagnostic": 0,
        "loss_formulation_diagnostic": 1,
        "diagnostic_epochs": diagnostic_epochs(task.epochs),
        "diagnostic_stages": 0,
        "uniform_collapse_diagnostic": 0,
        "diagnostic_output_dir": str(run_dir),
        "diagnostic_only_first_epoch": 1,
        "output_dir": str(run_dir),
        "eval_every": 1,
    }
    command = [args.python_bin, "-u", "edge_main.py"]
    for name, value in values.items():
        command.extend([f"--{name}", str(value)])
    return command


def validate_command(command: list[str], task: Task) -> None:
    parsed = {command[index][2:]: command[index + 1] for index in range(3, len(command), 2)}
    required = dict(FIXED)
    required.update({"lambda_esg": task.lambda_esg, "lambda_prox": task.lambda_prox, "epoch": task.epochs})
    for key, expected in required.items():
        actual = parsed.get(key)
        if actual is None:
            raise RuntimeError(f"Missing required flag --{key}")
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"Flag mismatch --{key}: actual={actual}, expected={expected}")
        elif str(actual) != str(expected):
            raise RuntimeError(f"Flag mismatch --{key}: actual={actual}, expected={expected}")
    if float(parsed["lambda_orth"]) != 0.0 or float(parsed["global_orth_scale"]) != 0.0:
        raise RuntimeError("Orth must be fully disabled")


def gpu_inventory() -> list[dict]:
    probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    result = []
    for line in probe.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 6:
            result.append(
                {
                    "index": fields[0], "name": fields[1], "uuid": fields[2],
                    "memory_total_mb": fields[3], "memory_used_mb": fields[4], "utilization_gpu": fields[5],
                }
            )
    return result


def cache_audit(args) -> dict:
    root_text = str(args.root_dir)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from edge_data import load_edge_event_data
    from utils import hash_cfg

    audit = {}
    for dataset in DATASETS:
        data = load_edge_event_data(str(args.asset_root / "dataset"), dataset)
        pi_cfg = {
            "dataset": dataset,
            "method": "temporal_state_forest",
            "alpha": 0.2,
            "T": 4,
            "forest_samples": 50,
            "edge_neighbor_k": -1,
            "edge_ppr_topk": 20,
            "beta": 5.0,
            "seed": 42,
            "num_events": int(data.num_events),
            "forest_impl": "state_expanded_temporal_subdivision_forest",
        }
        pi_hash = hash_cfg(pi_cfg)
        affinity_cfg = dict(pi_cfg)
        affinity_cfg.update(
            {
                "affinity_sparsify": "symmetric_union_knn_v1",
                "affinity_sparsify_effective": "symmetric_union_knn_v1",
            }
        )
        affinity_hash = hash_cfg(affinity_cfg)
        cache_dir = args.asset_root / "cache" / dataset
        paths = {
            "transition": cache_dir / f"edge_transition_{pi_hash}.npz",
            "Pi_E": cache_dir / f"edge_ppr_temporal_state_forest_{pi_hash}.npz",
            "Pi_cut": cache_dir / f"edge_ncut_affinity_temporal_state_forest_{affinity_hash}.npz",
        }
        entry = {
            "num_events": data.num_events,
            "pi_config_hash": pi_hash,
            "affinity_config_hash": affinity_hash,
            "paths": {name: str(path) for name, path in paths.items()},
            "hits": {name: path.exists() for name, path in paths.items()},
        }
        entry["all_hit"] = all(entry["hits"].values())
        audit[dataset] = entry
    return audit


def partition(all_tasks: list[Task], devices: list[str]) -> dict[str, list[Task]]:
    queues = {device: [] for device in devices}
    loads = {device: 0.0 for device in devices}
    for task in sorted(all_tasks, key=lambda item: item.estimated_cost, reverse=True):
        device = min(devices, key=lambda item: loads[item])
        queues[device].append(task)
        loads[device] += task.estimated_cost
    return queues


def write_csv(path: Path, rows: list[dict], fields=SUMMARY_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def best_metric(rows: list[dict], metric: str) -> tuple[object, object]:
    valid = [(number(row.get(metric)), row.get("epoch", "")) for row in rows]
    valid = [(value, epoch) for value, epoch in valid if math.isfinite(value)]
    if not valid:
        return "", ""
    return max(valid, key=lambda item: item[0])


def summary_row(output_dir: Path, task: Task, assigned_gpu: str = "") -> dict:
    run_dir = task_dir(output_dir, task)
    result = read_json(run_dir / "result.json")
    rows = read_csv(run_dir / "metrics.csv")
    final = rows[-1] if rows else {}
    best_acc, best_acc_epoch = best_metric(rows, "ACC")
    best_nmi, best_nmi_epoch = best_metric(rows, "NMI")
    best_ari, best_ari_epoch = best_metric(rows, "ARI")
    best_f1, best_f1_epoch = best_metric(rows, "Macro_F1")
    epoch_times = [number(row.get("epoch_seconds")) for row in rows]
    epoch_times = [value for value in epoch_times if math.isfinite(value)]
    prox_times = [number(row.get("prox_forward_backward_seconds")) for row in rows]
    prox_times = [value for value in prox_times if math.isfinite(value)]
    memories = [number(row.get("peak_gpu_memory_mb")) for row in rows]
    memories = [value for value in memories if math.isfinite(value)]
    error = result.get("error", "")
    return {
        "dataset": task.dataset,
        "lambda_esg": task.lambda_esg,
        "lambda_prox": task.lambda_prox,
        "epochs": task.epochs,
        "status": result.get("status", "missing"),
        "error": error,
        "final_ACC": final.get("ACC", ""),
        "final_NMI": final.get("NMI", ""),
        "final_ARI": final.get("ARI", ""),
        "final_Macro_F1": final.get("Macro_F1", ""),
        "best_ACC": best_acc, "best_ACC_epoch": best_acc_epoch,
        "best_NMI": best_nmi, "best_NMI_epoch": best_nmi_epoch,
        "best_ARI": best_ari, "best_ARI_epoch": best_ari_epoch,
        "best_Macro_F1": best_f1, "best_Macro_F1_epoch": best_f1_epoch,
        "final_cut": final.get("cut_loss", ""),
        "final_esg": final.get("esg_loss", ""),
        "final_prox": final.get("prox_loss", ""),
        "final_gain_mean": final.get("mean_gain", ""),
        "final_gain_min": final.get("min_gain", ""),
        "final_gain_max": final.get("max_gain", ""),
        "final_rank1": final.get("q_rank1_energy_ratio", ""),
        "final_effective_rank": final.get("q_effective_rank", ""),
        "final_q_margin": final.get("q_margin_mean", ""),
        "final_volume_cv": final.get("cluster_volume_cv", final.get("cluster_volume_coefficient_of_variation", "")),
        "final_qtdq_condition": final.get("qtdq_condition_number", ""),
        "final_uniform_l2": final.get("q_uniform_l2_mean", ""),
        "final_entropy_gap": final.get("q_entropy_gap", ""),
        "final_node_up_prox": final.get("node_update_from_prox", ""),
        "final_node_up_global": final.get("node_update_from_global", ""),
        "final_node_update_ratio": final.get("node_update_prox_global_ratio", ""),
        "mean_epoch_seconds": statistics.mean(epoch_times) if epoch_times else "",
        "mean_prox_seconds": statistics.mean(prox_times) if prox_times else "",
        "prox_optimizer_steps": final.get("prox_optimizer_steps", ""),
        "peak_gpu_memory_mb": max(memories) if memories else result.get("peak_gpu_memory_mb", ""),
        "runtime_seconds": result.get("runtime_seconds", ""),
        "assigned_physical_gpu": result.get("assigned_physical_gpu", assigned_gpu),
    }


def all_summary_rows(output_dir: Path, all_tasks: list[Task], assignments: dict[Task, str]) -> list[dict]:
    return [summary_row(output_dir, task, assignments.get(task, "")) for task in all_tasks]


def metric_rank(rows: list[dict], metric: str, row: dict) -> float:
    values = sorted((number(item.get(metric)) for item in rows), reverse=True)
    values = [value for value in values if math.isfinite(value)]
    value = number(row.get(metric))
    if not values or not math.isfinite(value):
        return float(len(rows) + 1)
    return 1.0 + sum(candidate > value for candidate in values)


def collapse_flags(row: dict) -> tuple[bool, bool]:
    rank1 = number(row.get("final_rank1"))
    effective_rank = number(row.get("final_effective_rank"))
    margin = number(row.get("final_q_margin"))
    entropy_gap = number(row.get("final_entropy_gap"))
    uniform_l2 = number(row.get("final_uniform_l2"))
    rank_collapse = (math.isfinite(rank1) and rank1 >= 0.98) or (
        math.isfinite(effective_rank) and effective_rank <= 1.2
    )
    uniform_collapse = (
        math.isfinite(margin) and math.isfinite(entropy_gap) and math.isfinite(uniform_l2)
        and margin <= 1e-4 and abs(entropy_gap) <= 1e-4 and uniform_l2 <= 1e-4
    )
    return rank_collapse, uniform_collapse


def write_analysis(path: Path, rows: list[dict]) -> None:
    successful = [row for row in rows if row.get("status") == "success"]
    lines = [
        "# ETGC Stage-1 Hyperparameter Search Without Orth",
        "",
        f"Completed: {len(successful)}/{len(rows)}. Orth is disabled in every run (`lambda_orth=0`, `global_orth_scale=0`).",
        "Best-epoch fields are diagnostic only; ranking below prioritizes final-epoch metrics.",
        "",
    ]
    if not successful:
        lines.append("No completed runs are available yet.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines += [
        "## Top 5 per dataset",
        "",
        "| Dataset | ESG | Prox | Final ACC | Final F1 | Final NMI | Final ARI | Rank1 | Eff. rank | Late F1 drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        group = [row for row in successful if row["dataset"] == dataset]
        group.sort(key=lambda row: number(row.get("final_Macro_F1")), reverse=True)
        for row in group[:5]:
            late_drop = number(row.get("best_Macro_F1")) - number(row.get("final_Macro_F1"))
            values = [
                dataset, row["lambda_esg"], row["lambda_prox"], row["final_ACC"], row["final_Macro_F1"],
                row["final_NMI"], row["final_ARI"], row["final_rank1"], row["final_effective_rank"], late_drop,
            ]
            rendered = [str(values[0])] + ["" if not math.isfinite(number(value)) else f"{number(value):.6f}" for value in values[1:]]
            lines.append("| " + " | ".join(rendered) + " |")

    combinations = []
    for esg in VALUES:
        for prox in VALUES:
            group = [row for row in successful if number(row["lambda_esg"]) == esg and number(row["lambda_prox"]) == prox]
            if len(group) != len(DATASETS):
                continue
            rank_scores = []
            rank_collapses = uniform_collapses = 0
            late_drops = []
            for row in group:
                dataset_rows = [item for item in successful if item["dataset"] == row["dataset"]]
                rank_scores.append(statistics.mean(metric_rank(dataset_rows, metric, row) for metric in (
                    "final_Macro_F1", "final_ACC", "final_NMI", "final_ARI"
                )))
                rank_flag, uniform_flag = collapse_flags(row)
                rank_collapses += int(rank_flag)
                uniform_collapses += int(uniform_flag)
                late_drops.append(max(0.0, number(row["best_Macro_F1"]) - number(row["final_Macro_F1"])))
            combinations.append(
                {
                    "lambda_esg": esg,
                    "lambda_prox": prox,
                    "mean_external_rank": statistics.mean(rank_scores),
                    "mean_final_f1": statistics.mean(number(row["final_Macro_F1"]) for row in group),
                    "rank_collapse_datasets": rank_collapses,
                    "uniform_collapse_datasets": uniform_collapses,
                    "mean_late_drop": statistics.mean(late_drops),
                }
            )
    combinations.sort(
        key=lambda row: (
            row["rank_collapse_datasets"], row["uniform_collapse_datasets"],
            row["mean_external_rank"], row["mean_late_drop"],
        )
    )
    lines += [
        "", "## Cross-dataset unified top 5", "",
        "| ESG | Prox | Mean external rank | Mean final F1 | Rank-collapse datasets | Uniform-collapse datasets | Mean late drop |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combinations[:5]:
        lines.append(
            f"| {row['lambda_esg']:.2f} | {row['lambda_prox']:.2f} | {row['mean_external_rank']:.3f} | "
            f"{row['mean_final_f1']:.6f} | {row['rank_collapse_datasets']} | "
            f"{row['uniform_collapse_datasets']} | {row['mean_late_drop']:.6f} |"
        )

    lines += ["", "## Diagnostics", ""]
    for dataset in DATASETS:
        group = [row for row in successful if row["dataset"] == dataset]
        rank_count = sum(collapse_flags(row)[0] for row in group)
        uniform_count = sum(collapse_flags(row)[1] for row in group)
        late_count = sum(
            number(row["best_Macro_F1"]) - number(row["final_Macro_F1"]) > 0.05 for row in group
        )
        lines.append(
            f"- **{dataset}**: rank-collapse={rank_count}/{len(group)}, "
            f"uniform-collapse={uniform_count}/{len(group)}, severe late degradation={late_count}/{len(group)}."
        )
    if len(combinations) == 25:
        selected = combinations[:5]
        lines += ["", "## Recommended candidates for stage 2", ""]
        for row in selected[:5]:
            lines.append(f"- `lambda_esg={row['lambda_esg']}`, `lambda_prox={row['lambda_prox']}`")
        lines += [
            "",
            "Selection rule: no collapse first, then the mean within-dataset rank across final ACC/Macro-F1/NMI/ARI, "
            "then lower best-to-final degradation. No label metric was used to stop training.",
        ]
    else:
        lines += ["", "Unified recommendations are deferred until all 25 parameter pairs complete on all four datasets."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summaries(output_dir: Path, all_tasks: list[Task], assignments: dict[Task, str]) -> None:
    rows = all_summary_rows(output_dir, all_tasks, assignments)
    write_csv(output_dir / "summary.csv", rows)
    write_analysis(output_dir / "analysis.md", rows)


def run_task(args, task: Task, physical_gpu: str) -> dict:
    run_dir = task_dir(args.output_dir, task)
    if task_complete(run_dir, task):
        cleanup_run_dir(run_dir)
        return {"task": task, "status": "skipped", "physical_gpu": physical_gpu, "exit_code": 0}
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, task, run_dir)
    validate_command(command, task)
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("OPENBLAS_NUM_THREADS", "4")
    started = time.time()
    error = ""
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write(f"assigned_physical_gpu={physical_gpu}\n")
        log.write("CUDA_DEVICE_ORDER=PCI_BUS_ID\n")
        log.write(f"CUDA_VISIBLE_DEVICES={physical_gpu}\n")
        log.write("resolved_command=" + " ".join(command) + "\n")
        log.flush()
        try:
            process = subprocess.run(command, cwd=args.root_dir, env=environment, stdout=log, stderr=subprocess.STDOUT)
            exit_code = process.returncode
        except Exception as exc:
            exit_code = 1
            error = f"{type(exc).__name__}: {exc}"
            log.write(error + "\n")
        log.write(f"runner_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"runner_exit_code={exit_code}\n")
    result_path = run_dir / "result.json"
    result = read_json(result_path)
    if exit_code != 0 or result.get("status") != "success":
        result.update(
            {
                "status": "failed", "error": error or f"edge_main exit_code={exit_code}",
                "assigned_physical_gpu": physical_gpu,
            }
        )
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        result["assigned_physical_gpu"] = physical_gpu
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cleanup_run_dir(run_dir)
    return {
        "task": task,
        "status": "success" if task_complete(run_dir, task) else "failed",
        "physical_gpu": physical_gpu,
        "exit_code": exit_code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--physical-gpus", default="1,2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve()
    args.asset_root = args.asset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    devices = [item.strip() for item in args.physical_gpus.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one physical GPU is required")
    all_tasks = tasks()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=args.root_dir, text=True, capture_output=True, check=True
    ).stdout.strip()
    if not branch:
        branch = "exp/etgc-mainline-refine (detached worktree at exact commit)"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.root_dir, text=True, capture_output=True, check=True
    ).stdout.strip()
    inventory = gpu_inventory()
    inventory_by_index = {item["index"]: item for item in inventory}
    missing_devices = [device for device in devices if device not in inventory_by_index]
    if missing_devices:
        raise RuntimeError(f"Unknown physical GPUs: {missing_devices}")
    cache = cache_audit(args)
    missing_cache = [dataset for dataset, entry in cache.items() if not entry["all_hit"]]
    if missing_cache:
        raise RuntimeError(f"Required exact Pi_E/Pi_cut cache is missing for: {missing_cache}; refusing grid launch")

    queues = partition(all_tasks, devices)
    assignments = {task: device for device, queue in queues.items() for task in queue}
    code_info = args.output_dir / "code_info"
    code_info.mkdir(exist_ok=True)
    (code_info / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    (code_info / "environment.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(), "python_bin": args.python_bin,
                "branch": branch, "commit": commit, "gpu_inventory": inventory,
                "selected_physical_gpus": devices,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (code_info / "cache_audit.json").write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (code_info / "common_config.json").write_text(
        json.dumps(
            {
                "method": "ETGC", "purpose": "stage-1 lambda_esg x lambda_prox search without orth",
                "datasets": DATASETS, "epochs": EPOCHS, "lambda_esg_values": VALUES,
                "lambda_prox_values": VALUES, "fixed": FIXED, "task_count": len(all_tasks),
                "branch": branch, "commit": commit,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    plan_rows = [
        {
            "dataset": task.dataset, "lambda_esg": task.lambda_esg, "lambda_prox": task.lambda_prox,
            "epochs": task.epochs, "assigned_physical_gpu": assignments[task],
            "run_dir": str(task_dir(args.output_dir, task)),
        }
        for task in all_tasks
    ]
    write_csv(args.output_dir / "experiment_plan.csv", plan_rows, tuple(plan_rows[0]))
    for task in all_tasks:
        validate_command(build_command(args, task, task_dir(args.output_dir, task)), task)

    print(f"branch={branch}", flush=True)
    print(f"commit={commit}", flush=True)
    print("gpu_inventory=" + json.dumps(inventory, sort_keys=True), flush=True)
    print("cache_hits=" + json.dumps({key: value["all_hit"] for key, value in cache.items()}, sort_keys=True), flush=True)
    print(f"task_count={len(all_tasks)}", flush=True)
    for device in devices:
        counts = {dataset: sum(task.dataset == dataset for task in queues[device]) for dataset in DATASETS}
        print(f"gpu_assignment physical={device} tasks={len(queues[device])} datasets={json.dumps(counts, sort_keys=True)}", flush=True)
    write_summaries(args.output_dir, all_tasks, assignments)
    if args.dry_run:
        print("dry_run=true all_commands_validated=true", flush=True)
        return 0

    started = time.time()
    summary_lock = threading.Lock()
    records = []

    def run_queue(device: str, queue: list[Task]) -> list[dict]:
        local_records = []
        for task in queue:
            print(
                f"run dataset={task.dataset} lambda_esg={task.lambda_esg} lambda_prox={task.lambda_prox} "
                f"epochs={task.epochs} physical_gpu={device}",
                flush=True,
            )
            record = run_task(args, task, device)
            local_records.append(record)
            print(
                f"done dataset={task.dataset} lambda_esg={task.lambda_esg} lambda_prox={task.lambda_prox} "
                f"status={record['status']} physical_gpu={device}",
                flush=True,
            )
            with summary_lock:
                write_summaries(args.output_dir, all_tasks, assignments)
        return local_records

    threads = []
    results_by_device = {}

    def worker(device: str) -> None:
        results_by_device[device] = run_queue(device, queues[device])

    for device in devices:
        thread = threading.Thread(target=worker, args=(device,), name=f"gpu-{device}")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    for device in devices:
        records.extend(results_by_device.get(device, []))
    write_summaries(args.output_dir, all_tasks, assignments)
    elapsed = time.time() - started
    failures = [record for record in records if record["status"] == "failed"]
    (args.output_dir / "run_complete.json").write_text(
        json.dumps(
            {
                "status": "success" if not failures else "completed_with_failures",
                "task_count": len(all_tasks), "failure_count": len(failures),
                "total_runtime_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"summary={args.output_dir / 'summary.csv'}", flush=True)
    print(f"analysis={args.output_dir / 'analysis.md'}", flush=True)
    print(f"total_runtime_seconds={elapsed:.6f}", flush=True)
    print(f"failure_count={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
