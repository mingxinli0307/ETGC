#!/usr/bin/env python3
"""Evaluate ETGC immediately after component-aware KMeans prior initialization."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from run_l3_multiseed_validation import cleanup, command, read_json, stages_for
from run_paper_objective_validation import set_flag, validate_devices, visible_cuda_device_names


DATASETS = ("school", "dblp", "patent", "arXivAI")
SEEDS = (42, 43, 44, 45, 46)
BASE_OVERRIDES = {
    "require_pretrained_node2vec": 1,
    "node_prior_mode": "component_structural",
    "node_prior_restarts": 500,
    "node_prior_bisecting_restarts": 50,
    "node_prior_logit_strength": 4.0,
    "node_prior_event_role": "source",
    "lambda_node_prior": 0.0,
    "global_cut_scale": 1.0,
    "global_orth_scale": 1.0,
}


@dataclass(frozen=True)
class Task:
    dataset: str
    seed: int


def make_tasks() -> list[Task]:
    return [Task(dataset, seed) for dataset in DATASETS for seed in SEEDS]


def run_dir_for(output_dir: Path, task: Task) -> Path:
    return output_dir / task.dataset / f"seed{task.seed}"


def build_task_command(args, task: Task, run_dir: Path) -> list[str]:
    worker = SimpleNamespace(
        device=args.device,
        asset_root=args.asset_root,
        python_bin=args.python_bin,
    )
    cmd = command(
        worker,
        task.seed,
        run_dir,
        prototype_seed=task.seed,
        epoch=1,
        init_only=True,
        dataset=task.dataset,
        cluster_loss_type="matrix_ncut",
        orth_type="orth",
        forest_samples=50,
        cluster_head_type="legacy_mlp",
    )
    for name, value in BASE_OVERRIDES.items():
        set_flag(cmd, name, value)
    set_flag(cmd, "node_prior_seed", task.seed)
    set_flag(cmd, "diagnostic_epochs", "")
    return cmd


def task_complete(run_dir: Path) -> bool:
    result = read_json(run_dir / "result.json")
    config = read_json(run_dir / "config.json")
    metrics_path = run_dir / "metrics.csv"
    if result.get("status") != "success" or not metrics_path.exists():
        return False
    rows = max(0, len(metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    return (
        rows >= 1
        and int(config.get("init_only", 0)) == 1
        and int(result.get("best_epoch", -1)) == 0
        and bool(result.get("final_metrics"))
    )


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def stage_value(stage: dict, key: str):
    aliases = {"cluster_volume_cv": "cluster_volume_coefficient_of_variation"}
    if key in stage:
        return stage[key]
    return stage.get(aliases.get(key, ""), "")


def result_row(output_dir: Path, task: Task) -> dict:
    run_dir = run_dir_for(output_dir, task)
    result = read_json(run_dir / "result.json")
    config = read_json(run_dir / "config.json")
    stages = stages_for(run_dir)
    init = stages.get("after_cluster_initialization") or stages.get("after_prototype_initialization") or {}
    metrics = result.get("final_metrics", {})
    prior_info = result.get("node_prior_info", config.get("node_prior_info", {})) or {}
    return {
        "dataset": task.dataset,
        "seed": task.seed,
        "status": result.get("status", "missing"),
        "evaluation_stage": "after_cluster_initialization",
        "init_only": config.get("init_only", ""),
        "ACC": metrics.get("ACC", ""),
        "Macro_F1": metrics.get("Macro_F1", ""),
        "NMI": metrics.get("NMI", ""),
        "ARI": metrics.get("ARI", ""),
        "q_rank1_energy_ratio": stage_value(init, "q_rank1_energy_ratio"),
        "q_centered_to_total_energy_ratio": stage_value(init, "q_centered_to_total_energy_ratio"),
        "q_effective_rank": stage_value(init, "q_effective_rank"),
        "q_normalized_margin_mean": stage_value(init, "q_normalized_margin_mean"),
        "cluster_volume_cv": stage_value(init, "cluster_volume_cv"),
        "num_active_edge_clusters": stage_value(init, "num_active_edge_clusters"),
        "num_active_node_clusters": stage_value(init, "num_active_node_clusters"),
        "largest_edge_cluster_ratio": stage_value(init, "largest_edge_cluster_ratio"),
        "largest_node_cluster_ratio": stage_value(init, "largest_node_cluster_ratio"),
        "node_prior_mode_effective": prior_info.get("node_prior_mode_effective", ""),
        "node_prior_connected_components": prior_info.get("node_prior_connected_components", ""),
        "node_prior_selected_inertia": prior_info.get("node_prior_selected_inertia", ""),
        "initialization_seed": config.get("model_seed", task.seed),
        "prototype_seed": config.get("prototype_seed", task.seed),
        "node_prior_seed": config.get("node_prior_seed", task.seed),
        "runtime_seconds": result.get("runtime_seconds", ""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    metrics = (
        "ACC",
        "Macro_F1",
        "NMI",
        "ARI",
        "q_rank1_energy_ratio",
        "q_centered_to_total_energy_ratio",
        "q_effective_rank",
        "q_normalized_margin_mean",
        "cluster_volume_cv",
        "largest_edge_cluster_ratio",
        "largest_node_cluster_ratio",
        "runtime_seconds",
    )
    output = []
    for dataset in DATASETS:
        group = [row for row in rows if row["dataset"] == dataset and row["status"] == "success"]
        aggregate = {
            "dataset": dataset,
            "successful_seeds": len(group),
            "expected_seeds": len(SEEDS),
        }
        for metric in metrics:
            values = [number(row.get(metric)) for row in group]
            values = [value for value in values if value is not None]
            aggregate[f"{metric}_mean"] = statistics.mean(values) if values else ""
            aggregate[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "")
        output.append(aggregate)
    return output


def write_report(path: Path, aggregates: list[dict]) -> None:
    lines = [
        "# ETGC Component-aware KMeans Prior Initialization Evaluation",
        "",
        "No optimization epoch or global loss update is executed. Metrics are computed immediately after",
        "component-aware node prior construction and cluster/prototype initialization, using the standard Q -> S evaluation.",
        "",
        "| Dataset | ACC | Macro-F1 | NMI | ARI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        values = []
        for metric in ("ACC", "Macro_F1", "NMI", "ARI"):
            mean = number(row.get(f"{metric}_mean"))
            std = number(row.get(f"{metric}_std"))
            values.append("" if mean is None else f"{100.0 * mean:.3f} +/- {100.0 * std:.3f}")
        lines.append(f"| {row['dataset']} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "All values are mean +/- population standard deviation over seeds 42-46.",
            "ACC and Macro-F1 use Hungarian label matching.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summaries(output_dir: Path, tasks: list[Task]) -> None:
    rows = [result_row(output_dir, task) for task in tasks]
    write_csv(output_dir / "initialization_summary.csv", rows)
    aggregates = aggregate_rows(rows)
    write_csv(output_dir / "initialization_aggregate.csv", aggregates)
    write_report(output_dir / "diagnosis_report.md", aggregates)


def write_code_info(args, tasks: list[Task], visible_names: list[str]) -> None:
    code_info = args.output_dir / "code_info"
    code_info.mkdir(exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.root_dir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    (code_info / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    environment = {
        "python": args.python_bin,
        "platform": platform.platform(),
        "device": args.device,
        "visible_cuda_device_names": visible_names,
    }
    (code_info / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    common = {
        "method": "ETGC",
        "purpose": "component-aware KMeans prior initialization-only evaluation",
        "datasets": DATASETS,
        "seeds": SEEDS,
        "evaluation_stage": "after_cluster_initialization",
        "optimization_epochs_executed": 0,
        "base_overrides": BASE_OVERRIDES,
        "cluster_loss_type": "matrix_ncut",
        "orth_type": "orth",
        "forest_samples": 50,
        "task_count": len(tasks),
        "actual_commit": commit,
    }
    (code_info / "common_config.json").write_text(
        json.dumps(common, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_task(args, task: Task) -> dict:
    run_dir = run_dir_for(args.output_dir, task)
    if not args.no_resume and task_complete(run_dir):
        cleanup(run_dir)
        return {"task": task, "status": "skipped", "exit_code": 0}
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_task_command(args, task, run_dir)
    started = time.time()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("command=" + " ".join(cmd) + "\n")
        process = subprocess.run(cmd, cwd=args.root_dir, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nrunner_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"runner_exit_code={process.returncode}\n")
    cleanup(run_dir)
    return {
        "task": task,
        "status": "success" if process.returncode == 0 else "failed",
        "exit_code": process.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-device-name-contains", default="A100")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve()
    args.asset_root = args.asset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    visible_names = visible_cuda_device_names(args.python_bin)
    validate_devices([args.device], visible_names, args.require_device_name_contains)
    tasks = make_tasks()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_code_info(args, tasks, visible_names)
    write_csv(
        args.output_dir / "experiment_plan.csv",
        [
            {
                "dataset": task.dataset,
                "seed": task.seed,
                "evaluation_stage": "after_cluster_initialization",
                "init_only": 1,
                "overrides": json.dumps(BASE_OVERRIDES, sort_keys=True),
            }
            for task in tasks
        ],
    )
    if args.dry_run:
        for task in tasks:
            print(" ".join(build_task_command(args, task, run_dir_for(args.output_dir, task))))
        print(f"task_count={len(tasks)}")
        return 0
    failures = []
    for task in tasks:
        print(f"run dataset={task.dataset} seed={task.seed} device={args.device}", flush=True)
        record = run_task(args, task)
        print(f"done dataset={task.dataset} seed={task.seed} status={record['status']}", flush=True)
        if record["exit_code"] != 0:
            failures.append(record)
        write_summaries(args.output_dir, tasks)
    print(f"task_count={len(tasks)} failures={len(failures)}")
    print(f"initialization_summary={args.output_dir / 'initialization_summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
