#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

from run_l3_multiseed_validation import command, complete, cleanup, number, read_csv, read_json, stages_for, stage_value


DATASETS = ["school", "dblp", "patent", "arXivAI"]
SEEDS = [42, 43]
FOREST_SAMPLES = 50


def fmt(value):
    value = number(value)
    return "nan" if math.isnan(value) else f"{value:.6g}"


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_one(args, dataset, seed, run_dir):
    cmd = command(
        args,
        seed,
        run_dir,
        prototype_seed=seed,
        epoch=20,
        init_only=False,
        dataset=dataset,
        cluster_loss_type="matrix_ncut",
        orth_type=args.orth_type,
        forest_samples=FOREST_SAMPLES,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("cmd=" + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=args.root_dir, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nscript_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"script_exit_code={proc.returncode}\n")
    cleanup(run_dir)
    if proc.returncode != 0 and not (run_dir / "result.json").exists():
        (run_dir / "result.json").write_text(json.dumps({
            "status": "failed", "dataset": dataset, "seed": seed, "exit_code": proc.returncode,
        }, indent=2, sort_keys=True), encoding="utf-8")
    return proc.returncode


def result_row(dataset, seed, run_dir):
    result = read_json(run_dir / "result.json")
    config = read_json(run_dir / "config.json")
    metrics = read_csv(run_dir / "metrics.csv")
    stages = stages_for(run_dir)
    init = stages.get("after_cluster_initialization") or stages.get("after_prototype_initialization") or {}
    final = stages.get("final_epoch") or stages.get("epoch_20") or {}
    valid = [row for row in metrics if math.isfinite(number(row.get("Macro_F1")))]
    best = max(valid, key=lambda row: number(row.get("Macro_F1")), default={})
    best_f1 = result.get("best_metrics", {}).get("Macro_F1", best.get("Macro_F1", ""))
    final_f1 = result.get("final_metrics", {}).get("Macro_F1", stage_value(final, "Macro_F1"))
    conditions = [number(row.get("qtdq_condition_number")) for row in valid]
    conditions = [value for value in conditions if math.isfinite(value)]
    finite_flags = [str(row.get("matrix_ncut_solve_finite", "")).lower() for row in valid]
    peak_memory = max([number(row.get("peak_gpu_memory_mb"), 0.0) for row in metrics] or [0.0])
    return {
        "dataset": dataset,
        "seed": seed,
        "cluster_loss_type": config.get("cluster_loss_type", "matrix_ncut"),
        "orth_type": config.get("orth_type", ""),
        "forest_samples": config.get("forest_samples", FOREST_SAMPLES),
        "M": result.get("M", config.get("M", "")),
        "N": result.get("N", config.get("N", "")),
        "K": result.get("K", config.get("K", "")),
        "initial_macro_f1": stage_value(init, "Macro_F1"),
        "best_macro_f1": best_f1,
        "best_epoch": result.get("best_epoch", best.get("epoch", "")),
        "final_macro_f1": final_f1,
        "best_to_final_drop": number(best_f1) - number(final_f1),
        "final_acc": result.get("final_metrics", {}).get("ACC", stage_value(final, "ACC")),
        "final_nmi": result.get("final_metrics", {}).get("NMI", stage_value(final, "NMI")),
        "final_ari": result.get("final_metrics", {}).get("ARI", stage_value(final, "ARI")),
        "initial_rank1_energy": stage_value(init, "q_rank1_energy_ratio"),
        "final_rank1_energy": stage_value(final, "q_rank1_energy_ratio"),
        "initial_center_ratio": stage_value(init, "q_centered_to_total_energy_ratio"),
        "final_center_ratio": stage_value(final, "q_centered_to_total_energy_ratio"),
        "initial_effective_rank": stage_value(init, "q_effective_rank"),
        "final_effective_rank": stage_value(final, "q_effective_rank"),
        "initial_normalized_margin": stage_value(init, "q_normalized_margin_mean"),
        "final_normalized_margin": stage_value(final, "q_normalized_margin_mean"),
        "initial_volume_cv": stage_value(init, "cluster_volume_cv"),
        "final_volume_cv": stage_value(final, "cluster_volume_cv"),
        "final_active_edge_clusters": stage_value(final, "num_active_edge_clusters"),
        "final_active_node_clusters": stage_value(final, "num_active_node_clusters"),
        "final_largest_edge_ratio": stage_value(final, "largest_edge_cluster_ratio"),
        "final_largest_node_ratio": stage_value(final, "largest_node_cluster_ratio"),
        "initial_qtdq_condition_number": stage_value(init, "qtdq_condition_number"),
        "max_qtdq_condition_number": max(conditions) if conditions else "",
        "all_matrix_solves_finite": bool(finite_flags) and all(flag == "true" for flag in finite_flags),
        "runtime_seconds": result.get("runtime_seconds", ""),
        "peak_gpu_memory_mb": peak_memory,
        "status": result.get("status", "missing"),
    }


def write_report(output_dir, rows, datasets, orth_type):
    dataset_scope = ", ".join(datasets)
    report = [
        f"# ETGC Matrix-Ncut + {orth_type} Forest-50 Cross-Dataset Validation", "",
        f"Scope: {dataset_scope}; seeds 42/43; 20 epochs; fixed C6; "
        f"matrix_ncut + {orth_type}; forest_samples=50; no legacy trace-ratio runs.", "",
        "| Dataset | Seed | K | Initial F1 | Best F1 | Best epoch | Final F1 | NMI | ARI | Final Rank1 | Center ratio | Eff rank | Norm margin | Edge active | Node active | QTDQ max | Runtime s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['dataset']} | {row['seed']} | {row['K']} | {fmt(row['initial_macro_f1'])} | "
            f"{fmt(row['best_macro_f1'])} | {row['best_epoch']} | {fmt(row['final_macro_f1'])} | "
            f"{fmt(row['final_nmi'])} | {fmt(row['final_ari'])} | {fmt(row['final_rank1_energy'])} | "
            f"{fmt(row['final_center_ratio'])} | {fmt(row['final_effective_rank'])} | "
            f"{fmt(row['final_normalized_margin'])} | {row['final_active_edge_clusters']} | "
            f"{row['final_active_node_clusters']} | {fmt(row['max_qtdq_condition_number'])} | "
            f"{fmt(row['runtime_seconds'])} |"
        )
    report += ["", "## Per-Dataset Aggregate", ""]
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset and row["status"] == "success"]
        if not subset:
            report.append(f"- {dataset}: no successful runs.")
            continue
        finals = [number(row["final_macro_f1"]) for row in subset]
        report.append(
            f"- {dataset}: mean final F1={fmt(statistics.mean(finals))}, "
            f"range={fmt(max(finals) - min(finals))}, min={fmt(min(finals))}, max={fmt(max(finals))}."
        )
    report += [
        "", "## Interpretation Boundary", "",
        "This is a direct transfer check of the provisional School L3 configuration. "
        "It is not dataset-specific tuning and must not be read as a final all-dataset ETGC result.",
    ]
    (output_dir / "diagnosis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--orth-type", choices=("orth", "orthqa"), default="orth")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.asset_root = args.asset_root.resolve()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    if not datasets:
        parser.error("--datasets must contain at least one dataset name")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_info = args.output_dir / "code_info"
    code_info.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.root_dir, capture_output=True, text=True).stdout.strip()
    log_line = subprocess.run(["git", "log", "-1", "--oneline"], cwd=args.root_dir, capture_output=True, text=True).stdout.strip()
    (code_info / "commit.txt").write_text(commit + "\n" + log_line + "\n", encoding="utf-8")
    (code_info / "environment.txt").write_text(
        f"python={args.python_bin}\ndevice={args.device}\n"
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
        f"TEMPORAL_FOREST_WORKERS={os.environ.get('TEMPORAL_FOREST_WORKERS', '')}\n"
        f"TEMPORAL_FOREST_CHUNK_SAMPLES={os.environ.get('TEMPORAL_FOREST_CHUNK_SAMPLES', '')}\n"
        f"TEMPORAL_FOREST_COMBINE_CHUNKS={os.environ.get('TEMPORAL_FOREST_COMBINE_CHUNKS', '')}\n",
        encoding="utf-8",
    )
    common = {
        "method": "ETGC", "purpose": f"matrix-Ncut + {args.orth_type} forest-50 cross-dataset validation",
        "datasets": datasets, "seeds": SEEDS, "epoch": 20,
        "forest_samples": FOREST_SAMPLES,
        "cluster_loss_type": "matrix_ncut", "orth_type": args.orth_type,
        "legacy_trace_ratio_enabled_in_experiment": False,
        "model_seed_equals_prototype_seed": True, "fixed_c6_configuration": True,
        "asset_root": str(args.asset_root), "save_embeddings": 0,
        "checkpoint": "disabled/not produced by edge_main.py",
    }
    (code_info / "common_config.json").write_text(json.dumps(common, indent=2, sort_keys=True), encoding="utf-8")

    failed = []
    for dataset in datasets:
        for seed in SEEDS:
            run_dir = args.output_dir / dataset / f"seed{seed}"
            if not args.no_resume and complete(run_dir):
                print(f"skip dataset={dataset} seed={seed}", flush=True)
                cleanup(run_dir)
                continue
            print(f"run dataset={dataset} seed={seed}", flush=True)
            if run_one(args, dataset, seed, run_dir) != 0:
                failed.append((dataset, seed))
    rows = [result_row(dataset, seed, args.output_dir / dataset / f"seed{seed}") for dataset in datasets for seed in SEEDS]
    fields = list(rows[0].keys())
    write_csv(args.output_dir / "cross_dataset_comparison.csv", fields, rows)
    write_report(args.output_dir, rows, datasets, args.orth_type)
    print(f"run_failures={failed}")
    print(f"comparison={args.output_dir / 'cross_dataset_comparison.csv'}")
    print(f"report={args.output_dir / 'diagnosis_report.md'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
