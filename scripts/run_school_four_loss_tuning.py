#!/usr/bin/env python3
"""Tune the four ETGC losses on School and summarize runtime composition.

Only matrix Ncut, ESG, cosine proximity, and OrthQA are allowed to contribute
to optimization.  The task list is a fixed, label-independent local design
around the best completed no-orth School configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


EPOCHS = 100
SEED = 42
ORTH_TYPE = "orthqa"
ALLOWED_LOSSES = ("matrix_ncut", "esg", "proximity", "orthqa")

# The anchor is the best final-Macro-F1 School point from the completed
# no-orth search: ncut=0.5, esg=0, prox=1, orth=0.  Each additional axis is
# perturbed around a low, non-zero OrthQA coefficient (0.1).  Large OrthQA
# coefficients are intentionally excluded because the completed 200--2000
# sweep already showed monotonic degradation.
ORTH_VALUES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
NCUT_VALUES = (0.1, 0.25, 1.0, 2.0)
ANCHOR_EFFECTIVE_ORTH_WEIGHT = 0.5 * 0.1
PROX_VALUES = (0.0, 0.25, 0.5, 1.5, 2.0)
ESG_VALUES = (0.02, 0.05, 0.1, 0.25)

FIXED = {
    "dataset": "school",
    "directed": 0,
    "batch_size": 512,
    "epoch": EPOCHS,
    "learning_rate": 1e-4,
    "node_emb_lr": 1e-5,
    "edge_dim": 128,
    "time_dim": 32,
    "edge_hidden_dim": 128,
    "cluster_hidden_dim": 64,
    "prototype_temperature": 0.2,
    "alpha": 0.2,
    "beta": 5.0,
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
    "orth_type": ORTH_TYPE,
    "global_cut_scale": 1.0,
    "global_orth_scale": 1.0,
    "seed": SEED,
    "model_seed": SEED,
    "prototype_seed": SEED,
    "forest_seed": SEED,
    "cluster_init_mode": "random",
    "prototype_init_mode": "random",
    "node_emb_mode": "small_lr",
    "cluster_output_bias_mode": "default",
    "cluster_input_norm": "none",
    "node_prior_mode": "none",
    "lambda_proj": 0.0,
    "lambda_bal": 0.0,
    "lambda_node_anchor": 0.0,
    "lambda_node_sbm": 0.0,
    "lambda_node_prior": 0.0,
    "direct_kmeans_eval": 0,
    "direct_node_prior_eval": 0,
    "save_embeddings": 0,
    "global_q_chunk_size": 8192,
    "global_ncut_row_block_size": 65536,
    "quiet": 1,
    "overnight_diagnostic": 0,
    "loss_formulation_diagnostic": 1,
    "diagnostic_epochs": "1,5,10,20,30,50,75,100",
    "diagnostic_stages": 0,
    "uniform_collapse_diagnostic": 0,
    "diagnostic_only_first_epoch": 1,
    "eval_every": 1,
}

SUMMARY_FIELDS = (
    "config", "axis", "seed", "epochs", "status", "error",
    "lambda_edge_ncut", "lambda_esg", "lambda_prox", "lambda_orth",
    "effective_orth_weight", "best_epoch", "best_ACC", "best_Macro_F1",
    "best_NMI", "best_ARI", "final_ACC", "final_Macro_F1", "final_NMI",
    "final_ARI", "final_cut", "final_esg", "final_prox", "final_orth",
    "final_weighted_cut", "final_weighted_esg", "final_weighted_prox",
    "final_weighted_orth", "final_total_global", "final_rank1",
    "final_effective_rank", "final_q_margin", "final_uniform_l2",
    "final_entropy_gap", "edge_active", "edge_largest_ratio", "node_active",
    "node_largest_ratio", "runtime_seconds", "mean_epoch_seconds",
    "median_epoch_seconds", "mean_prox_seconds", "mean_global_forward_seconds",
    "mean_global_backward_seconds", "mean_residual_seconds", "prox_time_fraction",
    "mean_prox_pair_build_seconds", "mean_prox_pair_build_wait_seconds",
    "mean_prox_h2d_seconds", "mean_prox_encode_seconds",
    "mean_prox_loss_forward_seconds", "mean_prox_backward_seconds",
    "mean_prox_optimizer_step_seconds", "mean_prox_pair_count",
    "mean_prox_optimizer_steps",
    "measured_global_loss_time_fraction", "peak_gpu_memory_mb", "assigned_physical_gpu",
    "run_dir",
)


def value_slug(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Task:
    name: str
    axis: str
    lambda_edge_ncut: float
    lambda_esg: float
    lambda_prox: float
    lambda_orth: float

    @property
    def slug(self) -> str:
        return (
            f"{self.name}_ncut_{value_slug(self.lambda_edge_ncut)}"
            f"_esg_{value_slug(self.lambda_esg)}"
            f"_prox_{value_slug(self.lambda_prox)}"
            f"_orth_{value_slug(self.lambda_orth)}"
        )

    @property
    def estimated_cost(self) -> float:
        return 1.0 if self.lambda_prox == 0.0 else 8.0


def make_tasks() -> list[Task]:
    tasks: list[Task] = []
    for value in ORTH_VALUES:
        tasks.append(Task("orth", "orth", 0.5, 0.0, 1.0, value))
    for value in NCUT_VALUES:
        # In the current implementation lambda_edge_ncut multiplies both the
        # cut and relative OrthQA term.  Hold their product fixed on this axis
        # so changing Ncut weight does not silently change absolute OrthQA.
        tasks.append(
            Task(
                "ncut",
                "ncut",
                value,
                0.0,
                1.0,
                ANCHOR_EFFECTIVE_ORTH_WEIGHT / value,
            )
        )
    for value in PROX_VALUES:
        tasks.append(Task("prox", "prox", 0.5, 0.0, value, 0.1))
    for value in ESG_VALUES:
        tasks.append(Task("esg", "esg", 0.5, value, 1.0, 0.1))
    unique: dict[tuple[float, float, float, float], Task] = {}
    for task in tasks:
        key = (task.lambda_edge_ncut, task.lambda_esg, task.lambda_prox, task.lambda_orth)
        unique.setdefault(key, task)
    return list(unique.values())


def number(value, default=math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


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


def run_dir(output_dir: Path, task: Task) -> Path:
    return output_dir / task.slug / f"seed{SEED}"


def complete(output_dir: Path, task: Task) -> bool:
    directory = run_dir(output_dir, task)
    result = read_json(directory / "result.json")
    config = read_json(directory / "config.json")
    rows = read_csv(directory / "metrics.csv")
    if result.get("status") != "success" or len(rows) < EPOCHS:
        return False
    expected = {
        "dataset": "school",
        "epoch": EPOCHS,
        "cluster_loss_type": "matrix_ncut",
        "orth_type": ORTH_TYPE,
        "lambda_edge_ncut": task.lambda_edge_ncut,
        "lambda_esg": task.lambda_esg,
        "lambda_prox": task.lambda_prox,
        "lambda_orth": task.lambda_orth,
        "lambda_proj": 0.0,
        "lambda_bal": 0.0,
        "lambda_node_anchor": 0.0,
        "lambda_node_sbm": 0.0,
        "lambda_node_prior": 0.0,
    }
    for key, expected_value in expected.items():
        actual = config.get(key)
        if isinstance(expected_value, float):
            if not math.isclose(number(actual), expected_value, abs_tol=1e-12, rel_tol=0.0):
                return False
        elif actual != expected_value:
            return False
    return int(number(rows[-1].get("epoch"), -1)) == EPOCHS


def build_command(args, task: Task, directory: Path) -> list[str]:
    values = {
        **FIXED,
        "device": "cuda:0",
        "lambda_edge_ncut": task.lambda_edge_ncut,
        "lambda_esg": task.lambda_esg,
        "lambda_prox": task.lambda_prox,
        "lambda_orth": task.lambda_orth,
        "data_root": str(args.asset_root / "dataset"),
        "emb_root": str(args.asset_root / "emb"),
        "pretrain_emb_dir": str(args.asset_root / "pretrain"),
        "cache_dir": str(args.asset_root / "cache"),
        "diagnostic_output_dir": str(directory),
        "output_dir": str(directory),
    }
    command = [args.python_bin, "-u", "edge_main.py"]
    for key, value in values.items():
        command.extend([f"--{key}", str(value)])
    return command


def validate_command(command: list[str], task: Task) -> None:
    parsed = {command[index][2:]: command[index + 1] for index in range(3, len(command), 2)}
    expected = {
        "dataset": "school",
        "cluster_loss_type": "matrix_ncut",
        "orth_type": ORTH_TYPE,
        "lambda_edge_ncut": task.lambda_edge_ncut,
        "lambda_esg": task.lambda_esg,
        "lambda_prox": task.lambda_prox,
        "lambda_orth": task.lambda_orth,
        "lambda_proj": 0.0,
        "lambda_bal": 0.0,
        "lambda_node_anchor": 0.0,
        "lambda_node_sbm": 0.0,
        "lambda_node_prior": 0.0,
    }
    for key, expected_value in expected.items():
        actual = parsed.get(key)
        if actual is None:
            raise RuntimeError(f"Missing --{key}")
        if isinstance(expected_value, float):
            if not math.isclose(float(actual), expected_value, abs_tol=1e-12, rel_tol=0.0):
                raise RuntimeError(f"Unexpected --{key}={actual}; expected {expected_value}")
        elif actual != str(expected_value):
            raise RuntimeError(f"Unexpected --{key}={actual}; expected {expected_value}")


def best(rows: list[dict], metric: str) -> tuple[float, object]:
    values = [(number(row.get(metric)), row.get("epoch", "")) for row in rows]
    values = [item for item in values if math.isfinite(item[0])]
    return max(values, default=(math.nan, ""), key=lambda item: item[0])


def mean_field(rows: list[dict], field: str) -> float:
    values = [number(row.get(field)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return statistics.mean(values) if values else math.nan


def summary_row(output_dir: Path, task: Task, gpu: str = "") -> dict:
    directory = run_dir(output_dir, task)
    result = read_json(directory / "result.json")
    rows = read_csv(directory / "metrics.csv")
    final = rows[-1] if rows else {}
    best_f1, best_epoch = best(rows, "Macro_F1")
    best_acc, _ = best(rows, "ACC")
    best_nmi, _ = best(rows, "NMI")
    best_ari, _ = best(rows, "ARI")
    epoch_times = [number(row.get("epoch_seconds")) for row in rows]
    epoch_times = [value for value in epoch_times if math.isfinite(value)]
    mean_epoch = statistics.mean(epoch_times) if epoch_times else math.nan
    median_epoch = statistics.median(epoch_times) if epoch_times else math.nan
    mean_prox = mean_field(rows, "prox_forward_backward_seconds")
    mean_forward = mean_field(rows, "cluster_forward_seconds")
    mean_backward = mean_field(rows, "cluster_backward_seconds")
    mean_residual = mean_epoch - mean_prox - mean_forward - mean_backward
    memories = [number(row.get("peak_gpu_memory_mb")) for row in rows]
    memories = [value for value in memories if math.isfinite(value)]
    return {
        "config": task.slug,
        "axis": task.axis,
        "seed": SEED,
        "epochs": EPOCHS,
        "status": result.get("status", "missing"),
        "error": result.get("error", ""),
        "lambda_edge_ncut": task.lambda_edge_ncut,
        "lambda_esg": task.lambda_esg,
        "lambda_prox": task.lambda_prox,
        "lambda_orth": task.lambda_orth,
        "effective_orth_weight": task.lambda_edge_ncut * task.lambda_orth,
        "best_epoch": best_epoch,
        "best_ACC": best_acc,
        "best_Macro_F1": best_f1,
        "best_NMI": best_nmi,
        "best_ARI": best_ari,
        "final_ACC": final.get("ACC", ""),
        "final_Macro_F1": final.get("Macro_F1", ""),
        "final_NMI": final.get("NMI", ""),
        "final_ARI": final.get("ARI", ""),
        "final_cut": final.get("cut_loss", ""),
        "final_esg": final.get("esg_loss", ""),
        "final_prox": final.get("prox_loss", ""),
        "final_orth": final.get("selected_penalty_loss", final.get("orth_loss", "")),
        "final_weighted_cut": final.get("weighted_cut_loss", ""),
        "final_weighted_esg": task.lambda_esg * number(final.get("esg_loss"), 0.0),
        "final_weighted_prox": final.get("weighted_proximity_loss", ""),
        "final_weighted_orth": final.get("weighted_orth_loss", ""),
        "final_total_global": final.get("global_total_loss", ""),
        "final_rank1": final.get("q_rank1_energy_ratio", ""),
        "final_effective_rank": final.get("q_effective_rank", ""),
        "final_q_margin": final.get("q_margin_mean", ""),
        "final_uniform_l2": final.get("q_uniform_l2_mean", ""),
        "final_entropy_gap": final.get("q_entropy_gap", ""),
        "edge_active": final.get("edge_hard_active_clusters", ""),
        "edge_largest_ratio": final.get("edge_hard_largest_ratio", ""),
        "node_active": final.get("node_hard_active_clusters", ""),
        "node_largest_ratio": final.get("node_hard_largest_ratio", ""),
        "runtime_seconds": result.get("runtime_seconds", final.get("total_runtime_seconds", "")),
        "mean_epoch_seconds": mean_epoch,
        "median_epoch_seconds": median_epoch,
        "mean_prox_seconds": mean_prox,
        "mean_global_forward_seconds": mean_forward,
        "mean_global_backward_seconds": mean_backward,
        "mean_residual_seconds": mean_residual,
        "prox_time_fraction": mean_prox / mean_epoch if mean_epoch > 0 else math.nan,
        "mean_prox_pair_build_seconds": mean_field(rows, "prox_pair_build_seconds"),
        "mean_prox_pair_build_wait_seconds": mean_field(rows, "prox_pair_build_wait_seconds"),
        "mean_prox_h2d_seconds": mean_field(rows, "prox_h2d_seconds"),
        "mean_prox_encode_seconds": mean_field(rows, "prox_encode_seconds"),
        "mean_prox_loss_forward_seconds": mean_field(rows, "prox_loss_forward_seconds"),
        "mean_prox_backward_seconds": mean_field(rows, "prox_backward_seconds"),
        "mean_prox_optimizer_step_seconds": mean_field(rows, "prox_optimizer_step_seconds"),
        "mean_prox_pair_count": mean_field(rows, "prox_pair_count"),
        "mean_prox_optimizer_steps": mean_field(rows, "prox_optimizer_steps"),
        "measured_global_loss_time_fraction": (
            (mean_forward + mean_backward) / mean_epoch if mean_epoch > 0 else math.nan
        ),
        "peak_gpu_memory_mb": max(memories) if memories else "",
        "assigned_physical_gpu": result.get("assigned_physical_gpu", gpu),
        "run_dir": str(directory),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def collapsed(row: dict) -> bool:
    rank1 = number(row.get("final_rank1"))
    effective_rank = number(row.get("final_effective_rank"))
    edge_active = number(row.get("edge_active"))
    node_active = number(row.get("node_active"))
    return rank1 >= 0.98 or effective_rank <= 1.2 or edge_active <= 1 or node_active <= 1


def write_analysis(path: Path, rows: list[dict]) -> None:
    successful = [row for row in rows if row.get("status") == "success"]
    ranked = sorted(successful, key=lambda row: (collapsed(row), -number(row.get("final_Macro_F1"))))
    lines = [
        "# School Four-Loss Tuning",
        "",
        f"Completed: {len(successful)}/{len(rows)}.",
        "Only matrix Ncut, ESG, cosine proximity, and OrthQA can contribute to optimization.",
        "All other loss coefficients are explicitly zero.",
        "Best-epoch metrics are diagnostic only; ranking uses final Macro-F1 after excluding collapse.",
        "",
        "## Top configurations",
        "",
        "| Config | Ncut | ESG | Prox | Orth | ACC | Macro-F1 | NMI | ARI | Rank1 | EffRank | Runtime min | Epoch s | Prox share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked[:10]:
        lines.append(
            f"| {row['config']} | {number(row['lambda_edge_ncut']):.3g} | {number(row['lambda_esg']):.3g} | "
            f"{number(row['lambda_prox']):.3g} | {number(row['lambda_orth']):.3g} | "
            f"{number(row['final_ACC']):.6f} | {number(row['final_Macro_F1']):.6f} | "
            f"{number(row['final_NMI']):.6f} | {number(row['final_ARI']):.6f} | "
            f"{number(row['final_rank1']):.6f} | {number(row['final_effective_rank']):.4f} | "
            f"{number(row['runtime_seconds']) / 60.0:.2f} | {number(row['mean_epoch_seconds']):.3f} | "
            f"{100.0 * number(row['prox_time_fraction']):.1f}% |"
        )
    lines += [
        "",
        "## Runtime",
        "",
        "The proximity substage columns use CUDA events without synchronizing every batch. "
        "Pair-build work is prefetched; pair wait is the portion exposed on the training critical path.",
        "",
        "| Config | Epoch s | Prox s | Pair wait s | H2D s | Encode s | Loss forward s | Prox backward s | Adam step s | Prox share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked[:10]:
        lines.append(
            f"| {row['config']} | {number(row['mean_epoch_seconds']):.3f} | "
            f"{number(row['mean_prox_seconds']):.3f} | "
            f"{number(row['mean_prox_pair_build_wait_seconds']):.3f} | "
            f"{number(row['mean_prox_h2d_seconds']):.3f} | "
            f"{number(row['mean_prox_encode_seconds']):.3f} | "
            f"{number(row['mean_prox_loss_forward_seconds']):.3f} | "
            f"{number(row['mean_prox_backward_seconds']):.3f} | "
            f"{number(row['mean_prox_optimizer_step_seconds']):.3f} | "
            f"{100.0 * number(row['prox_time_fraction']):.1f}% |"
        )
    lines += [""]
    if successful:
        with_prox = [row for row in successful if number(row["lambda_prox"]) > 0]
        without_prox = [row for row in successful if number(row["lambda_prox"]) == 0]
        for label, group in (("proximity enabled", with_prox), ("proximity disabled", without_prox)):
            if not group:
                continue
            lines.append(
                f"- {label}: mean runtime={statistics.mean(number(row['runtime_seconds']) for row in group) / 60.0:.2f} min, "
                f"mean epoch={statistics.mean(number(row['mean_epoch_seconds']) for row in group):.3f} s, "
                f"mean proximity share={100.0 * statistics.mean(number(row['prox_time_fraction']) for row in group):.1f}%."
            )
    if ranked:
        winner = ranked[0]
        lines += [
            "", "## Recommendation", "",
            f"Best non-collapsed final configuration: Ncut={winner['lambda_edge_ncut']}, "
            f"ESG={winner['lambda_esg']}, Prox={winner['lambda_prox']}, Orth={winner['lambda_orth']} "
            f"(ACC={number(winner['final_ACC']):.6f}, Macro-F1={number(winner['final_Macro_F1']):.6f}, "
            f"NMI={number(winner['final_NMI']):.6f}, ARI={number(winner['final_ARI']):.6f}).",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def partition(tasks: list[Task], devices: list[str]) -> dict[str, list[Task]]:
    queues = {device: [] for device in devices}
    loads = {device: 0.0 for device in devices}
    for task in sorted(tasks, key=lambda item: item.estimated_cost, reverse=True):
        device = min(devices, key=lambda item: loads[item])
        queues[device].append(task)
        loads[device] += task.estimated_cost
    return queues


def gpu_compute_process_count(physical_gpu: str) -> int:
    gpu_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    uuid = ""
    for row in gpu_rows:
        fields = [field.strip() for field in row.split(",", 1)]
        if len(fields) == 2 and fields[0] == str(physical_gpu):
            uuid = fields[1]
            break
    if not uuid:
        raise RuntimeError(f"Physical GPU {physical_gpu} is not visible to nvidia-smi")
    process_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return sum(1 for row in process_rows if row.split(",", 1)[0].strip() == uuid)


def wait_until_gpu_idle(
    physical_gpu: str,
    poll_seconds: float,
    stable_checks: int,
    stop_event: Optional[threading.Event] = None,
) -> bool:
    consecutive = 0
    checks = 0
    while consecutive < stable_checks:
        if stop_event is not None and stop_event.is_set():
            return False
        count = gpu_compute_process_count(physical_gpu)
        checks += 1
        if count == 0:
            consecutive += 1
        else:
            consecutive = 0
        if checks == 1 or checks % 10 == 0 or consecutive > 0:
            print(
                f"gpu_wait physical_gpu={physical_gpu} compute_processes={count} "
                f"idle_checks={consecutive}/{stable_checks}",
                flush=True,
            )
        if consecutive < stable_checks:
            if stop_event is not None and stop_event.wait(poll_seconds):
                return False
            if stop_event is None:
                time.sleep(poll_seconds)
    return True


def run_task(args, task: Task, gpu: str) -> dict:
    directory = run_dir(args.output_dir, task)
    directory.mkdir(parents=True, exist_ok=True)
    if complete(args.output_dir, task):
        return {"task": task, "gpu": gpu, "status": "skipped", "exit_code": 0}
    command = build_command(args, task, directory)
    validate_command(command, task)
    (directory / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    started = time.time()
    with (directory / "train.log").open("w", encoding="utf-8") as stream:
        stream.write(f"assigned_physical_gpu={gpu}\n")
        stream.write("resolved_command=" + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.run(command, cwd=args.root_dir, env=env, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"\nrunner_runtime_seconds={time.time() - started:.6f}\n")
        stream.write(f"runner_exit_code={process.returncode}\n")
    result_path = directory / "result.json"
    result = read_json(result_path)
    result["assigned_physical_gpu"] = str(gpu)
    if process.returncode != 0 or result.get("status") != "success":
        result["status"] = "failed"
        result.setdefault("error", f"edge_main exit_code={process.returncode}")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"task": task, "gpu": gpu, "status": result.get("status"), "exit_code": process.returncode}


def refresh(output_dir: Path, tasks: list[Task], assignments: dict[Task, str]) -> list[dict]:
    rows = [summary_row(output_dir, task, assignments.get(task, "")) for task in tasks]
    write_csv(output_dir / "summary.csv", rows)
    write_analysis(output_dir / "analysis.md", rows)
    return rows


def main() -> int:
    global EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--dependency-complete-file", type=Path)
    parser.add_argument("--dependency-pid", type=int, default=0)
    parser.add_argument("--dependency-poll-seconds", type=float, default=60.0)
    parser.add_argument("--wait-for-free-gpus", action="store_true")
    parser.add_argument("--gpu-poll-seconds", type=float, default=60.0)
    parser.add_argument("--gpu-stable-checks", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve()
    args.asset_root = args.asset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    EPOCHS = int(args.epochs)
    if EPOCHS <= 0:
        raise ValueError("epochs must be positive")
    FIXED["epoch"] = EPOCHS
    diagnostic_epochs = [1, 5, 10, 20, 30, 50, 75, 100]
    if EPOCHS >= 150:
        diagnostic_epochs.append(150)
    if EPOCHS >= 200:
        diagnostic_epochs.append(200)
    FIXED["diagnostic_epochs"] = ",".join(
        str(epoch) for epoch in diagnostic_epochs if epoch <= EPOCHS
    )
    devices = [item.strip() for item in args.physical_gpus.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one physical GPU is required")
    if args.gpu_poll_seconds <= 0 or args.gpu_stable_checks <= 0:
        raise ValueError("GPU polling interval and stable-check count must be positive")
    if args.dependency_poll_seconds <= 0:
        raise ValueError("dependency polling interval must be positive")
    if not (args.asset_root / "dataset" / "school" / "school.txt").exists():
        raise FileNotFoundError("School dataset is missing from asset root")

    tasks = make_tasks()
    queues = partition(tasks, devices)
    assignments = {task: device for device, queue in queues.items() for task in queue}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_info = args.output_dir / "code_info"
    code_info.mkdir(exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.root_dir, text=True, capture_output=True, check=True
    ).stdout.strip()
    common = {
        "method": "ETGC",
        "dataset": "school",
        "seed": SEED,
        "epochs": EPOCHS,
        "allowed_losses": ALLOWED_LOSSES,
        "task_count": len(tasks),
        "task_design": {
            "orth_values": ORTH_VALUES,
            "ncut_values_at_fixed_effective_orth_weight": NCUT_VALUES,
            "ncut_axis_effective_orth_weight": ANCHOR_EFFECTIVE_ORTH_WEIGHT,
            "prox_values_at_orth_0p1": PROX_VALUES,
            "esg_values_at_orth_0p1": ESG_VALUES,
        },
        "fixed": FIXED,
        "commit": commit,
        "physical_gpus": devices,
        "scheduling": "dynamic_shared_queue",
    }
    (code_info / "common_config.json").write_text(
        json.dumps(common, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (code_info / "environment.json").write_text(
        json.dumps({"platform": platform.platform(), "python": args.python_bin}, indent=2) + "\n",
        encoding="utf-8",
    )
    plan_rows = [
        {
            "config": task.slug,
            "axis": task.axis,
            "lambda_edge_ncut": task.lambda_edge_ncut,
            "lambda_esg": task.lambda_esg,
            "lambda_prox": task.lambda_prox,
            "lambda_orth": task.lambda_orth,
            "initial_queue_gpu": assignments[task],
        }
        for task in tasks
    ]
    with (args.output_dir / "task_plan.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plan_rows[0]))
        writer.writeheader()
        writer.writerows(plan_rows)
    if args.dry_run:
        for task in tasks:
            command = build_command(args, task, run_dir(args.output_dir, task))
            validate_command(command, task)
            print(assignments[task], " ".join(command))
        print(f"task_count={len(tasks)}")
        return 0

    if args.dependency_complete_file is not None:
        dependency_path = args.dependency_complete_file.resolve()
        while not dependency_path.exists():
            if args.dependency_pid > 0 and not Path(f"/proc/{args.dependency_pid}").exists():
                raise RuntimeError(
                    f"Dependency process {args.dependency_pid} ended without {dependency_path}"
                )
            print(
                f"dependency_wait file={dependency_path} pid={args.dependency_pid}",
                flush=True,
            )
            time.sleep(args.dependency_poll_seconds)
        dependency = read_json(dependency_path)
        if dependency.get("status") != "success":
            raise RuntimeError(
                f"Dependency run did not succeed: file={dependency_path} status={dependency.get('status')}"
            )
        print(f"dependency_success file={dependency_path}", flush=True)

    lock = threading.Lock()
    pending = sorted(tasks, key=lambda item: item.estimated_cost, reverse=True)
    all_assigned = threading.Event()

    def worker(device: str) -> list[dict]:
        records = []
        while True:
            if args.wait_for_free_gpus and not wait_until_gpu_idle(
                device,
                args.gpu_poll_seconds,
                args.gpu_stable_checks,
                stop_event=all_assigned,
            ):
                break
            with lock:
                if not pending:
                    break
                task = pending.pop(0)
                assignments[task] = device
                if not pending:
                    all_assigned.set()
            print(
                f"run config={task.slug} gpu={device} ncut={task.lambda_edge_ncut} "
                f"esg={task.lambda_esg} prox={task.lambda_prox} orth={task.lambda_orth}",
                flush=True,
            )
            record = run_task(args, task, device)
            records.append(record)
            print(f"done config={task.slug} status={record['status']} gpu={device}", flush=True)
            with lock:
                refresh(args.output_dir, tasks, assignments)
        return records

    started = time.time()
    records = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device) for device in devices]
        for future in as_completed(futures):
            records.extend(future.result())
    rows = refresh(args.output_dir, tasks, assignments)
    failures = [row for row in rows if row.get("status") != "success"]
    complete_count = sum(complete(args.output_dir, task) for task in tasks)
    run_complete = {
        "status": "success" if complete_count == len(tasks) and not failures else "failed",
        "task_count": len(tasks),
        "complete_count": complete_count,
        "failure_count": len(failures),
        "total_runtime_seconds": time.time() - started,
    }
    (args.output_dir / "run_complete.json").write_text(
        json.dumps(run_complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_complete, sort_keys=True), flush=True)
    return 0 if run_complete["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
