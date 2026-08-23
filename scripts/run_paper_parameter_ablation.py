#!/usr/bin/env python3
"""Run ETGC paper parameter sensitivity and final-configuration ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from run_l3_multiseed_validation import cleanup, command, read_json, stages_for


SOURCE_COMMIT_EXPECTED = "14f30fa888646011dbb91b08b99b2f6b2ddf0088"
PARAMETER_DATASETS = ("dblp", "patent")
ABLATION_DATASETS = ("school", "dblp", "patent", "arXivAI")
PARAMETER_SEEDS = (42,)
ABLATION_SEEDS = (42, 43)
DATASET_COST = {"arXivAI": 14.0, "dblp": 10.0, "school": 1.5, "patent": 1.0}
BASE_OVERRIDES = {
    "require_pretrained_node2vec": 1,
    "node_prior_mode": "component_structural",
    "node_prior_restarts": 500,
    "node_prior_bisecting_restarts": 50,
    "node_prior_logit_strength": 4.0,
    "node_prior_event_role": "source",
    "lambda_node_prior": 0.0,
}
PARAMETER_AXES = {
    "gamma": {
        "flag": "node_prior_logit_strength",
        "values": (0.0, 1.0, 2.0, 4.0, 8.0),
        "base": 4.0,
    },
    "alpha": {
        "flag": "alpha",
        "values": (0.1, 0.2, 0.3, 0.4, 0.5),
        "base": 0.2,
    },
    "lambda_orth": {
        "flag": "lambda_orth",
        "values": (0.0, 0.25, 0.5, 1.0, 2.0),
        "base": 1.0,
    },
}
ABLATIONS = {
    "A0_full": {},
    "A1_no_structural_prior": {"node_prior_logit_strength": 0.0},
    "A2_current_time_only": {"time_feature_mode": "current"},
    "A3_random_cluster_init": {"cluster_init_mode": "random"},
    "A4_no_layernorm": {"cluster_input_norm": "none"},
    "A5_no_global_cluster_objective": {"global_cut_scale": 0.0, "global_orth_scale": 0.0},
    "A6_no_orth": {"global_orth_scale": 0.0},
    "A7_global_prior": {"node_prior_mode": "global_structural"},
}


@dataclass(frozen=True)
class Task:
    family: str
    name: str
    dataset: str
    seed: int
    overrides: tuple[tuple[str, object], ...]

    @property
    def override_dict(self) -> dict:
        return dict(self.overrides)


def value_slug(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def set_flag(cmd: list[str], name: str, value: object) -> None:
    flag = f"--{name}"
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = str(value)
    else:
        cmd.extend([flag, str(value)])


def make_parameter_tasks() -> list[Task]:
    tasks = []
    for axis, spec in PARAMETER_AXES.items():
        for value in spec["values"]:
            name = f"{axis}_{value_slug(value)}"
            overrides = dict(BASE_OVERRIDES)
            overrides[spec["flag"]] = value
            for dataset in PARAMETER_DATASETS:
                for seed in PARAMETER_SEEDS:
                    tasks.append(Task("parameter", name, dataset, seed, tuple(overrides.items())))
    return tasks


def make_ablation_tasks() -> list[Task]:
    tasks = []
    for name, delta in ABLATIONS.items():
        datasets = ("patent",) if name == "A7_global_prior" else ABLATION_DATASETS
        overrides = dict(BASE_OVERRIDES)
        overrides.update(delta)
        for dataset in datasets:
            for seed in ABLATION_SEEDS:
                tasks.append(Task("ablation", name, dataset, seed, tuple(overrides.items())))
    return tasks


def partition_tasks(tasks: list[Task], device_count: int) -> list[list[Task]]:
    """Greedily balance expected dataset runtime while keeping one serial queue per GPU."""
    queues = [[] for _ in range(device_count)]
    loads = [0.0 for _ in range(device_count)]
    ordered = sorted(tasks, key=lambda task: DATASET_COST.get(task.dataset, 1.0), reverse=True)
    for task in ordered:
        index = min(range(device_count), key=lambda item: loads[item])
        queues[index].append(task)
        loads[index] += DATASET_COST.get(task.dataset, 1.0)
    return queues


def task_run_dir(output_dir: Path, task: Task) -> Path:
    return output_dir / task.family / task.name / task.dataset / f"seed{task.seed}"


def task_complete(run_dir: Path, epochs: int) -> bool:
    result = read_json(run_dir / "result.json")
    metrics = run_dir / "metrics.csv"
    config = read_json(run_dir / "config.json")
    if result.get("status") != "success" or not metrics.exists() or not config:
        return False
    rows = max(0, len(metrics.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    stages = stages_for(run_dir)
    return rows >= epochs and bool(stages.get("final_epoch") or stages.get(f"epoch_{epochs}"))


def build_task_command(args, task: Task, device: str, run_dir: Path) -> list[str]:
    worker_args = SimpleNamespace(**vars(args))
    worker_args.device = device
    cmd = command(
        worker_args,
        task.seed,
        run_dir,
        prototype_seed=task.seed,
        epoch=args.epochs,
        dataset=task.dataset,
        cluster_loss_type="matrix_ncut",
        orth_type="orth",
        forest_samples=50,
        cluster_head_type="legacy_mlp",
    )
    for name, value in task.override_dict.items():
        set_flag(cmd, name, value)
    set_flag(cmd, "node_prior_seed", task.seed)
    return cmd


def run_task(args, task: Task, device: str) -> dict:
    run_dir = task_run_dir(args.output_dir, task)
    if not args.no_resume and task_complete(run_dir, args.epochs):
        cleanup(run_dir)
        return {"task": task, "device": device, "status": "skipped", "exit_code": 0}
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_task_command(args, task, device, run_dir)
    started = time.time()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("command=" + " ".join(cmd) + "\n")
        log.write(f"task_family={task.family}\n")
        log.write(f"task_name={task.name}\n")
        log.write(f"assigned_device={device}\n")
        process = subprocess.run(cmd, cwd=args.root_dir, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nrunner_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"runner_exit_code={process.returncode}\n")
    cleanup(run_dir)
    return {
        "task": task,
        "device": device,
        "status": "success" if process.returncode == 0 else "failed",
        "exit_code": process.returncode,
    }


def final_diagnostic(run_dir: Path, epochs: int) -> dict:
    stages = stages_for(run_dir)
    return stages.get("final_epoch") or stages.get(f"epoch_{epochs}") or {}


def result_row(output_dir: Path, task: Task, epochs: int) -> dict:
    run_dir = task_run_dir(output_dir, task)
    result = read_json(run_dir / "result.json")
    config = read_json(run_dir / "config.json")
    final = result.get("final_metrics", {})
    diag = final_diagnostic(run_dir, epochs)
    parameter_name = ""
    parameter_value = ""
    if task.family == "parameter":
        parameter_name = task.name.rsplit("_", 1)[0]
        spec = PARAMETER_AXES[parameter_name]
        parameter_value = task.override_dict[spec["flag"]]
    return {
        "family": task.family,
        "config": task.name,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "dataset": task.dataset,
        "seed": task.seed,
        "status": result.get("status", "missing"),
        "final_macro_f1": final.get("Macro_F1", ""),
        "final_acc": final.get("ACC", ""),
        "final_nmi": final.get("NMI", ""),
        "final_ari": final.get("ARI", ""),
        "q_rank1_energy_ratio": diag.get("q_rank1_energy_ratio", ""),
        "q_centered_to_total_energy_ratio": diag.get("q_centered_to_total_energy_ratio", ""),
        "q_effective_rank": diag.get("q_effective_rank", ""),
        "q_normalized_margin_mean": diag.get("q_normalized_margin_mean", ""),
        "cluster_volume_cv": diag.get("cluster_volume_cv", diag.get("cluster_volume_coefficient_of_variation", "")),
        "num_active_edge_clusters": diag.get("num_active_edge_clusters", ""),
        "num_active_node_clusters": diag.get("num_active_node_clusters", ""),
        "largest_edge_cluster_ratio": diag.get("largest_edge_cluster_ratio", ""),
        "largest_node_cluster_ratio": diag.get("largest_node_cluster_ratio", ""),
        "qtdq_condition_number": diag.get("qtdq_condition_number", ""),
        "matrix_ncut_solve_finite": diag.get("matrix_ncut_solve_finite", ""),
        "runtime_seconds": result.get("runtime_seconds", ""),
        "peak_gpu_memory_mb": result.get("peak_gpu_memory_mb", ""),
        "alpha": config.get("alpha", ""),
        "node_prior_logit_strength": config.get("node_prior_logit_strength", ""),
        "lambda_orth": config.get("lambda_orth", ""),
        "lambda_edge_ncut": config.get("lambda_edge_ncut", ""),
        "global_cut_scale": config.get("global_cut_scale", 1.0),
        "global_orth_scale": config.get("global_orth_scale", 1.0),
        "time_feature_mode": config.get("time_feature_mode", ""),
        "cluster_init_mode": config.get("cluster_init_mode", ""),
        "cluster_input_norm": config.get("cluster_input_norm", ""),
        "node_prior_mode": config.get("node_prior_mode", ""),
        "node_prior_mode_effective": result.get("node_prior_info", {}).get("node_prior_mode_effective", ""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(output_dir: Path, tasks: list[Task], epochs: int) -> None:
    rows = [result_row(output_dir, task, epochs) for task in tasks]
    write_csv(output_dir / "combined_summary.csv", rows)
    write_csv(output_dir / "parameter_sensitivity.csv", [row for row in rows if row["family"] == "parameter"])
    write_csv(output_dir / "ablation_summary.csv", [row for row in rows if row["family"] == "ablation"])


def write_plan(output_dir: Path, tasks: list[Task]) -> None:
    fields = ["family", "config", "dataset", "seed", "overrides"]
    with (output_dir / "experiment_plan.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "family": task.family,
                    "config": task.name,
                    "dataset": task.dataset,
                    "seed": task.seed,
                    "overrides": json.dumps(task.override_dict, sort_keys=True),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="ETGC paper parameter sensitivity and ablation runner.")
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--mode", choices=("parameter", "ablation", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve()
    args.asset_root = args.asset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one device is required")

    tasks = []
    if args.mode in {"parameter", "all"}:
        tasks.extend(make_parameter_tasks())
    if args.mode in {"ablation", "all"}:
        tasks.extend(make_ablation_tasks())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plan(args.output_dir, tasks)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.root_dir, text=True, capture_output=True, check=True
    ).stdout.strip()
    code_info = args.output_dir / "code_info"
    code_info.mkdir(exist_ok=True)
    (code_info / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    environment = {
        "python": args.python_bin,
        "platform": platform.platform(),
        "devices": devices,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (code_info / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    common = {
        "method": "ETGC",
        "purpose": "paper parameter sensitivity and final-configuration ablation",
        "epochs": args.epochs,
        "parameter_datasets": PARAMETER_DATASETS,
        "parameter_seeds": PARAMETER_SEEDS,
        "parameter_axes": PARAMETER_AXES,
        "ablation_datasets": ABLATION_DATASETS,
        "ablation_seeds": ABLATION_SEEDS,
        "ablations": ABLATIONS,
        "base_overrides": BASE_OVERRIDES,
        "cluster_loss_type": "matrix_ncut",
        "orth_type": "orth",
        "forest_samples": 50,
        "source_commit_expected_when_designed": SOURCE_COMMIT_EXPECTED,
        "actual_commit": commit,
    }
    (code_info / "common_config.json").write_text(
        json.dumps(common, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.dry_run:
        for index, task in enumerate(tasks):
            device = devices[index % len(devices)]
            run_dir = task_run_dir(args.output_dir, task)
            print(f"{task.family},{task.name},{task.dataset},{task.seed},{device}")
            print(" ".join(build_task_command(args, task, device, run_dir)))
        print(f"task_count={len(tasks)}")
        return 0

    queues = partition_tasks(tasks, len(devices))

    def run_queue(device: str, queue: list[Task]) -> list[dict]:
        records = []
        for task in queue:
            print(f"run family={task.family} config={task.name} dataset={task.dataset} seed={task.seed} device={device}", flush=True)
            record = run_task(args, task, device)
            records.append(record)
            print(
                f"done family={task.family} config={task.name} dataset={task.dataset} "
                f"seed={task.seed} status={record['status']}",
                flush=True,
            )
        return records

    records = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(run_queue, device, queue) for device, queue in zip(devices, queues)]
        for future in as_completed(futures):
            records.extend(future.result())
    write_summaries(args.output_dir, tasks, args.epochs)
    failures = [record for record in records if record["exit_code"] != 0]
    print(f"task_count={len(tasks)} failures={len(failures)}")
    print(f"combined_summary={args.output_dir / 'combined_summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
