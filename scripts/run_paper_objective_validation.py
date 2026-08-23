#!/usr/bin/env python3
"""Run the final ETGC Cut x Orth attribution study on one fixed GPU type."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from run_l3_multiseed_validation import cleanup, command, read_json, stages_for


DATASETS = ("school", "dblp", "patent", "arXivAI")
SEEDS = (42, 43, 44, 45, 46)
DATASET_COST = {"arXivAI": 14.0, "dblp": 10.0, "school": 1.5, "patent": 1.0}
OBJECTIVES = {
    "O11_full": {"global_cut_scale": 1.0, "global_orth_scale": 1.0},
    "O10_cut_only": {"global_cut_scale": 1.0, "global_orth_scale": 0.0},
    "O01_orth_only": {"global_cut_scale": 0.0, "global_orth_scale": 1.0},
    "O00_prior_only": {"global_cut_scale": 0.0, "global_orth_scale": 0.0},
}
BASE_OVERRIDES = {
    "require_pretrained_node2vec": 1,
    "node_prior_mode": "component_structural",
    "node_prior_restarts": 500,
    "node_prior_bisecting_restarts": 50,
    "node_prior_logit_strength": 4.0,
    "node_prior_event_role": "source",
    "lambda_node_prior": 0.0,
}


@dataclass(frozen=True)
class Task:
    config: str
    dataset: str
    seed: int

    @property
    def overrides(self) -> dict:
        values = dict(BASE_OVERRIDES)
        values.update(OBJECTIVES[self.config])
        return values


def make_tasks() -> list[Task]:
    return [
        Task(config, dataset, seed)
        for config in OBJECTIVES
        for dataset in DATASETS
        for seed in SEEDS
    ]


def set_flag(cmd: list[str], name: str, value: object) -> None:
    flag = f"--{name}"
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = str(value)
    else:
        cmd.extend([flag, str(value)])


def run_dir_for(output_dir: Path, task: Task) -> Path:
    return output_dir / task.config / task.dataset / f"seed{task.seed}"


def task_complete(run_dir: Path, epochs: int) -> bool:
    result = read_json(run_dir / "result.json")
    config = read_json(run_dir / "config.json")
    metrics = run_dir / "metrics.csv"
    if result.get("status") != "success" or not config or not metrics.exists():
        return False
    metric_rows = max(0, len(metrics.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    stages = stages_for(run_dir)
    return metric_rows >= epochs and bool(stages.get("final_epoch") or stages.get(f"epoch_{epochs}"))


def build_task_command(args, task: Task, device: str, run_dir: Path) -> list[str]:
    worker = SimpleNamespace(**vars(args))
    worker.device = device
    cmd = command(
        worker,
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
    for name, value in task.overrides.items():
        set_flag(cmd, name, value)
    set_flag(cmd, "node_prior_seed", task.seed)
    return cmd


def partition_tasks(tasks: list[Task], device_count: int) -> list[list[Task]]:
    queues = [[] for _ in range(device_count)]
    loads = [0.0 for _ in range(device_count)]
    for task in sorted(tasks, key=lambda item: DATASET_COST[item.dataset], reverse=True):
        index = min(range(device_count), key=lambda item: loads[item])
        queues[index].append(task)
        loads[index] += DATASET_COST[task.dataset]
    return queues


def final_diagnostic(run_dir: Path, epochs: int) -> dict:
    stages = stages_for(run_dir)
    return stages.get("final_epoch") or stages.get(f"epoch_{epochs}") or {}


def result_row(output_dir: Path, task: Task, epochs: int) -> dict:
    run_dir = run_dir_for(output_dir, task)
    result = read_json(run_dir / "result.json")
    config = read_json(run_dir / "config.json")
    final = result.get("final_metrics", {})
    best = result.get("best_metrics", {})
    change = result.get("training_change_metrics", {})
    diag = final_diagnostic(run_dir, epochs)
    return {
        "config": task.config,
        "dataset": task.dataset,
        "seed": task.seed,
        "status": result.get("status", "missing"),
        "global_cut_scale": config.get("global_cut_scale", OBJECTIVES[task.config]["global_cut_scale"]),
        "global_orth_scale": config.get("global_orth_scale", OBJECTIVES[task.config]["global_orth_scale"]),
        "initial_acc": change.get("ACC_init", ""),
        "final_acc": final.get("ACC", ""),
        "acc_delta_final_init": change.get("ACC_delta_final_init", ""),
        "initial_macro_f1": change.get("MacroF1_init", ""),
        "best_macro_f1": best.get("Macro_F1", ""),
        "final_macro_f1": final.get("Macro_F1", ""),
        "macro_f1_delta_final_init": change.get("macro_f1_delta_final_init", ""),
        "initial_nmi": change.get("NMI_init", ""),
        "final_nmi": final.get("NMI", ""),
        "nmi_delta_final_init": change.get("NMI_delta_final_init", ""),
        "initial_ari": change.get("ARI_init", ""),
        "final_ari": final.get("ARI", ""),
        "ari_delta_final_init": change.get("ARI_delta_final_init", ""),
        "q_rank1_energy_ratio": diag.get("q_rank1_energy_ratio", ""),
        "q_centered_to_total_energy_ratio": diag.get("q_centered_to_total_energy_ratio", ""),
        "q_effective_rank": diag.get("q_effective_rank", ""),
        "q_normalized_margin_mean": diag.get("q_normalized_margin_mean", ""),
        "cluster_volume_cv": diag.get(
            "cluster_volume_cv", diag.get("cluster_volume_coefficient_of_variation", "")
        ),
        "num_active_edge_clusters": diag.get("num_active_edge_clusters", ""),
        "num_active_node_clusters": diag.get("num_active_node_clusters", ""),
        "largest_edge_cluster_ratio": diag.get("largest_edge_cluster_ratio", ""),
        "largest_node_cluster_ratio": diag.get("largest_node_cluster_ratio", ""),
        "qtdq_condition_number": diag.get("qtdq_condition_number", ""),
        "matrix_ncut_solve_finite": diag.get("matrix_ncut_solve_finite", ""),
        "q_drift_fro_normalized": change.get("q_drift_fro_normalized", ""),
        "cluster_weight_drift_relative": change.get("cluster_weight_drift_relative", ""),
        "edge_prediction_ari_init_final": change.get("edge_prediction_ari_init_final", ""),
        "node_prediction_ari_init_final": change.get("node_prediction_ari_init_final", ""),
        "runtime_seconds": result.get("runtime_seconds", ""),
        "peak_gpu_memory_mb": result.get("peak_gpu_memory_mb", ""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def aggregate_rows(rows: list[dict]) -> list[dict]:
    metrics = (
        "initial_acc",
        "final_acc",
        "acc_delta_final_init",
        "initial_macro_f1",
        "best_macro_f1",
        "final_macro_f1",
        "macro_f1_delta_final_init",
        "initial_nmi",
        "final_nmi",
        "initial_ari",
        "final_ari",
        "q_rank1_energy_ratio",
        "q_centered_to_total_energy_ratio",
        "q_effective_rank",
        "q_normalized_margin_mean",
        "cluster_volume_cv",
        "q_drift_fro_normalized",
        "cluster_weight_drift_relative",
        "edge_prediction_ari_init_final",
        "node_prediction_ari_init_final",
        "runtime_seconds",
        "peak_gpu_memory_mb",
    )
    output = []
    for config in OBJECTIVES:
        for dataset in DATASETS:
            group = [row for row in rows if row["config"] == config and row["dataset"] == dataset]
            aggregate = {
                "config": config,
                "dataset": dataset,
                "successful_seeds": sum(row["status"] == "success" for row in group),
                "expected_seeds": len(SEEDS),
            }
            for metric in metrics:
                values = [number(row.get(metric)) for row in group if row["status"] == "success"]
                values = [value for value in values if value is not None]
                aggregate[f"{metric}_mean"] = statistics.mean(values) if values else ""
                aggregate[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "")
            output.append(aggregate)
    return output


def write_report(path: Path, aggregates: list[dict]) -> None:
    lookup = {(row["config"], row["dataset"]): row for row in aggregates}
    lines = [
        "# ETGC Cut x Orth Objective Validation",
        "",
        "All values are mean over five seeds. ACC and Macro-F1 use Hungarian label matching.",
        "",
        "| Dataset | Config | ACC | Macro-F1 | NMI | ARI | F1 gain over prior-only | Train F1 delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        prior = number(lookup[("O00_prior_only", dataset)].get("final_macro_f1_mean"))
        for config in OBJECTIVES:
            row = lookup[(config, dataset)]
            final_f1 = number(row.get("final_macro_f1_mean"))
            gain = final_f1 - prior if final_f1 is not None and prior is not None else None
            values = [
                dataset,
                config,
                number(row.get("final_acc_mean")),
                final_f1,
                number(row.get("final_nmi_mean")),
                number(row.get("final_ari_mean")),
                gain,
                number(row.get("macro_f1_delta_final_init_mean")),
            ]
            rendered = [values[0], values[1]] + ["" if value is None else f"{value:.6f}" for value in values[2:]]
            lines.append("| " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "Interpretation rule: O11 must outperform O00 consistently before Matrix Ncut plus Orth can be claimed as a performance source.",
            "O10 vs O00 isolates Cut; O01 vs O00 isolates Orth; O11-O10-O01+O00 is the interaction term.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summaries(output_dir: Path, tasks: list[Task], epochs: int) -> None:
    rows = [result_row(output_dir, task, epochs) for task in tasks]
    write_csv(output_dir / "objective_summary.csv", rows)
    aggregates = aggregate_rows(rows)
    write_csv(output_dir / "objective_aggregate.csv", aggregates)
    write_report(output_dir / "diagnosis_report.md", aggregates)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visible_cuda_device_names(python_bin: str) -> list[str]:
    probe = subprocess.run(
        [
            python_bin,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps([torch.cuda.get_device_name(i) "
                "for i in range(torch.cuda.device_count())]))"
            ),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return list(json.loads(probe.stdout.strip()))


def validate_devices(devices: list[str], visible_names: list[str], required_substring: str) -> None:
    required = str(required_substring).strip().lower()
    for device in devices:
        if not device.startswith("cuda:"):
            raise ValueError(f"Formal objective validation requires CUDA, got {device}")
        index = int(device.split(":", 1)[1])
        if index >= len(visible_names):
            raise ValueError(f"Requested {device}, but only {len(visible_names)} CUDA devices are visible")
        if required and required not in visible_names[index].lower():
            raise RuntimeError(
                f"Requested {device} is {visible_names[index]!r}, which does not contain {required_substring!r}"
            )


def write_code_info(args, tasks: list[Task], devices: list[str], visible_names: list[str]) -> None:
    code_info = args.output_dir / "code_info"
    code_info.mkdir(exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.root_dir, text=True, capture_output=True, check=True
    ).stdout.strip()
    (code_info / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    environment = {
        "python": args.python_bin,
        "platform": platform.platform(),
        "devices": devices,
        "visible_cuda_device_names": visible_names,
        "required_device_name_substring": args.require_device_name_contains,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_inventory": gpu_query.stdout.strip().splitlines(),
    }
    (code_info / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    common = {
        "method": "ETGC",
        "purpose": "five-seed Cut x Orth attribution and prior-only validation",
        "datasets": DATASETS,
        "seeds": SEEDS,
        "epochs": args.epochs,
        "objectives": OBJECTIVES,
        "base_overrides": BASE_OVERRIDES,
        "cluster_loss_type": "matrix_ncut",
        "orth_type": "orth",
        "lambda_edge_ncut": 0.5,
        "lambda_orth": 1.0,
        "forest_samples": 50,
        "task_count": len(tasks),
        "actual_commit": commit,
    }
    (code_info / "common_config.json").write_text(
        json.dumps(common, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    embedding_audit = {}
    for dataset in DATASETS:
        path = args.asset_root / "pretrain" / f"{dataset}_feature.emb"
        embedding_audit[dataset] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    embedding_audit["source_audit"] = {
        "ground_truth_labels_used_by_pretrain_code": False,
        "input": "dataset temporal edge endpoints converted to an unweighted edge list",
        "note": "K is still obtained from the number of unique evaluation labels by ETGC.",
    }
    (code_info / "embedding_audit.json").write_text(
        json.dumps(embedding_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_task(args, task: Task, device: str) -> dict:
    run_dir = run_dir_for(args.output_dir, task)
    if not args.no_resume and task_complete(run_dir, args.epochs):
        cleanup(run_dir)
        return {"task": task, "device": device, "status": "skipped", "exit_code": 0}
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_task_command(args, task, device, run_dir)
    started = time.time()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("command=" + " ".join(cmd) + "\n")
        log.write(f"config={task.config}\nassigned_device={device}\n")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="ETGC five-seed Cut x Orth objective validation.")
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--require-device-name-contains", default="A100")
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
    visible_names = visible_cuda_device_names(args.python_bin)
    validate_devices(devices, visible_names, args.require_device_name_contains)
    tasks = make_tasks()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_code_info(args, tasks, devices, visible_names)
    write_csv(
        args.output_dir / "experiment_plan.csv",
        [
            {
                "config": task.config,
                "dataset": task.dataset,
                "seed": task.seed,
                "overrides": json.dumps(task.overrides, sort_keys=True),
            }
            for task in tasks
        ],
    )
    if args.dry_run:
        for index, task in enumerate(tasks):
            device = devices[index % len(devices)]
            print(" ".join(build_task_command(args, task, device, run_dir_for(args.output_dir, task))))
        print(f"task_count={len(tasks)}")
        return 0

    summary_lock = threading.Lock()
    queues = partition_tasks(tasks, len(devices))

    def run_queue(device: str, queue: list[Task]) -> list[dict]:
        records = []
        for task in queue:
            print(f"run config={task.config} dataset={task.dataset} seed={task.seed} device={device}", flush=True)
            record = run_task(args, task, device)
            records.append(record)
            print(
                f"done config={task.config} dataset={task.dataset} seed={task.seed} status={record['status']}",
                flush=True,
            )
            with summary_lock:
                write_summaries(args.output_dir, tasks, args.epochs)
        return records

    records = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(run_queue, device, queue) for device, queue in zip(devices, queues)]
        for future in as_completed(futures):
            records.extend(future.result())
    write_summaries(args.output_dir, tasks, args.epochs)
    failures = [record for record in records if record["exit_code"] != 0]
    print(f"task_count={len(tasks)} failures={len(failures)}")
    print(f"objective_summary={args.output_dir / 'objective_summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
