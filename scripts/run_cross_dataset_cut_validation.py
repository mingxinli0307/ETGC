#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from run_l3_multiseed_validation import command, complete, cleanup, number, read_csv, read_json, stages_for, stage_value


DATASETS = ["dblp", "patent", "arXivAI"]
SEEDS = [42, 43]
CONFIGS = [
    ("trace", "legacy_trace_ratio"),
    ("matrix", "matrix_ncut"),
]
ALLOWED = {"config.json", "metrics.csv", "diagnostic.json", "result.json", "train.log"}


def fmt(value):
    value = number(value)
    return "nan" if math.isnan(value) else f"{value:.6g}"


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def copy_matrix_reference(reference_dir, dataset, seed, run_dir):
    source = Path(reference_dir) / dataset / f"seed{seed}"
    if not complete(source):
        raise RuntimeError(f"Incomplete Matrix reference: {source}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ALLOWED:
        shutil.copy2(source / name, run_dir / name)
    cleanup(run_dir)


def run_trace(args, dataset, seed, run_dir):
    cmd = command(
        args, seed, run_dir, prototype_seed=seed, epoch=20, init_only=False,
        dataset=dataset, cluster_loss_type="legacy_trace_ratio", orth_type="orth",
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
            "status": "failed", "dataset": dataset, "seed": seed,
            "cluster_loss_type": "legacy_trace_ratio", "exit_code": proc.returncode,
        }, indent=2, sort_keys=True), encoding="utf-8")
    return proc.returncode


def summarize_run(config_name, loss_type, dataset, seed, run_dir):
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
    init_info = config.get("model_init_info", {}) or {}
    return {
        "config": config_name,
        "cluster_loss_type": loss_type,
        "orth_type": "orth",
        "dataset": dataset,
        "seed": seed,
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
        "final_rank1_energy": stage_value(final, "q_rank1_energy_ratio"),
        "final_center_ratio": stage_value(final, "q_centered_to_total_energy_ratio"),
        "final_effective_rank": stage_value(final, "q_effective_rank"),
        "final_normalized_margin": stage_value(final, "q_normalized_margin_mean"),
        "final_volume_cv": stage_value(final, "cluster_volume_cv"),
        "final_active_edge_clusters": stage_value(final, "num_active_edge_clusters"),
        "final_active_node_clusters": stage_value(final, "num_active_node_clusters"),
        "final_largest_edge_ratio": stage_value(final, "largest_edge_cluster_ratio"),
        "final_largest_node_ratio": stage_value(final, "largest_node_cluster_ratio"),
        "max_qtdq_condition_number": max(conditions) if conditions else "",
        "all_matrix_solves_finite": (
            bool(finite_flags) and all(flag == "true" for flag in finite_flags)
            if config_name == "matrix" else ""
        ),
        "runtime_seconds": result.get("runtime_seconds", ""),
        "peak_gpu_memory_mb": max([number(row.get("peak_gpu_memory_mb"), 0.0) for row in metrics] or [0.0]),
        "status": result.get("status", "missing"),
        "initial_q_checksum": init_info.get("initial_Q_summary_checksum", ""),
        "initial_weight_checksum": init_info.get("initial_cluster_weight_checksum", ""),
        "prototype_center_checksum": init_info.get("prototype_center_checksum", ""),
    }


def comparison_rows(rows):
    by_key = {(row["config"], row["dataset"], int(row["seed"])): row for row in rows}
    result = []
    for dataset in DATASETS:
        for seed in SEEDS:
            trace = by_key[("trace", dataset, seed)]
            matrix = by_key[("matrix", dataset, seed)]
            result.append({
                "dataset": dataset,
                "seed": seed,
                "trace_best_f1": trace["best_macro_f1"],
                "matrix_best_f1": matrix["best_macro_f1"],
                "matrix_minus_trace_best_f1": number(matrix["best_macro_f1"]) - number(trace["best_macro_f1"]),
                "trace_final_f1": trace["final_macro_f1"],
                "matrix_final_f1": matrix["final_macro_f1"],
                "matrix_minus_trace_final_f1": number(matrix["final_macro_f1"]) - number(trace["final_macro_f1"]),
                "trace_final_nmi": trace["final_nmi"],
                "matrix_final_nmi": matrix["final_nmi"],
                "trace_final_ari": trace["final_ari"],
                "matrix_final_ari": matrix["final_ari"],
                "trace_final_rank1": trace["final_rank1_energy"],
                "matrix_final_rank1": matrix["final_rank1_energy"],
                "trace_final_center_ratio": trace["final_center_ratio"],
                "matrix_final_center_ratio": matrix["final_center_ratio"],
                "trace_final_effective_rank": trace["final_effective_rank"],
                "matrix_final_effective_rank": matrix["final_effective_rank"],
                "trace_final_normalized_margin": trace["final_normalized_margin"],
                "matrix_final_normalized_margin": matrix["final_normalized_margin"],
                "trace_final_node_clusters": trace["final_active_node_clusters"],
                "matrix_final_node_clusters": matrix["final_active_node_clusters"],
                "initialization_checksums_match": all(
                    trace[key] == matrix[key] != ""
                    for key in ["initial_q_checksum", "initial_weight_checksum", "prototype_center_checksum"]
                ),
            })
    return result


def write_report(output_dir, rows, comparisons):
    report = [
        "# ETGC Cross-Dataset Cut Validation", "",
        "Fixed C6 and orth; only cut formulation changes. Matrix runs are reused from the completed L3 transfer check.", "",
        "| Dataset | Seed | Trace Best | Matrix Best | Trace Final | Matrix Final | Delta Final | Trace Rank1 | Matrix Rank1 | Trace Center | Matrix Center |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        report.append(
            f"| {row['dataset']} | {row['seed']} | {fmt(row['trace_best_f1'])} | "
            f"{fmt(row['matrix_best_f1'])} | {fmt(row['trace_final_f1'])} | "
            f"{fmt(row['matrix_final_f1'])} | {fmt(row['matrix_minus_trace_final_f1'])} | "
            f"{fmt(row['trace_final_rank1'])} | {fmt(row['matrix_final_rank1'])} | "
            f"{fmt(row['trace_final_center_ratio'])} | {fmt(row['matrix_final_center_ratio'])} |"
        )
    report += ["", "## Main Effects", ""]
    for dataset in DATASETS:
        subset = [row for row in comparisons if row["dataset"] == dataset]
        deltas = [number(row["matrix_minus_trace_final_f1"]) for row in subset]
        best_deltas = [number(row["matrix_minus_trace_best_f1"]) for row in subset]
        report.append(
            f"- {dataset}: mean Matrix-minus-Trace final F1={fmt(statistics.mean(deltas))}; "
            f"mean best-F1 delta={fmt(statistics.mean(best_deltas))}; "
            f"final direction consistent={all(delta > 0 for delta in deltas) or all(delta < 0 for delta in deltas)}."
        )
    checksum_ok = all(bool(row["initialization_checksums_match"]) for row in comparisons)
    matrix_wins = sum(number(row["matrix_minus_trace_final_f1"]) > 0 for row in comparisons)
    report += [
        "", "## Fairness and Decision", "",
        f"- Initialization checksum audit across cut formulations: {'PASS' if checksum_ok else 'FAIL'}.",
        f"- Matrix final F1 wins: {matrix_wins}/{len(comparisons)} dataset-seed pairs.",
        "- A universal Matrix objective requires consistent benefits beyond a single dataset and must also preserve assignment geometry.",
    ]
    (output_dir / "diagnosis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--matrix-reference-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.matrix_reference_dir = args.matrix_reference_dir.resolve()
    args.asset_root = args.asset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_info = args.output_dir / "code_info"
    code_info.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.root_dir, capture_output=True, text=True).stdout.strip()
    log_line = subprocess.run(["git", "log", "-1", "--oneline"], cwd=args.root_dir, capture_output=True, text=True).stdout.strip()
    (code_info / "commit.txt").write_text(commit + "\n" + log_line + "\n", encoding="utf-8")
    (code_info / "environment.txt").write_text(
        f"python={args.python_bin}\ndevice={args.device}\nCUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n",
        encoding="utf-8",
    )
    common = {
        "method": "ETGC", "purpose": "cross-dataset trace-ratio versus Matrix Ncut validation",
        "datasets": DATASETS, "seeds": SEEDS, "epoch": 20, "orth_type": "orth",
        "cut_formulations": ["legacy_trace_ratio", "matrix_ncut"],
        "new_runs": "legacy_trace_ratio only", "matrix_reference_dir": str(args.matrix_reference_dir),
        "fixed_c6_configuration": True, "model_seed_equals_prototype_seed": True,
        "asset_root": str(args.asset_root), "save_embeddings": 0,
        "checkpoint": "disabled/not produced by edge_main.py",
    }
    (code_info / "common_config.json").write_text(json.dumps(common, indent=2, sort_keys=True), encoding="utf-8")

    failed = []
    for dataset in DATASETS:
        for seed in SEEDS:
            matrix_dir = args.output_dir / "matrix" / dataset / f"seed{seed}"
            try:
                copy_matrix_reference(args.matrix_reference_dir, dataset, seed, matrix_dir)
            except Exception as exc:
                print(f"matrix_reuse_failed dataset={dataset} seed={seed} error={exc}", flush=True)
                failed.append(("matrix", dataset, seed))
            trace_dir = args.output_dir / "trace" / dataset / f"seed{seed}"
            if not args.no_resume and complete(trace_dir):
                print(f"skip trace dataset={dataset} seed={seed}", flush=True)
                cleanup(trace_dir)
            else:
                print(f"run trace dataset={dataset} seed={seed}", flush=True)
                if run_trace(args, dataset, seed, trace_dir) != 0:
                    failed.append(("trace", dataset, seed))

    rows = [
        summarize_run(config, loss_type, dataset, seed, args.output_dir / config / dataset / f"seed{seed}")
        for config, loss_type in CONFIGS for dataset in DATASETS for seed in SEEDS
    ]
    summary_fields = list(rows[0].keys())
    write_csv(args.output_dir / "combined_summary.csv", summary_fields, rows)
    comparisons = comparison_rows(rows)
    comparison_fields = list(comparisons[0].keys())
    write_csv(args.output_dir / "cut_comparison.csv", comparison_fields, comparisons)
    write_report(args.output_dir, rows, comparisons)
    print(f"run_failures={failed}")
    print(f"combined_summary={args.output_dir / 'combined_summary.csv'}")
    print(f"cut_comparison={args.output_dir / 'cut_comparison.csv'}")
    print(f"report={args.output_dir / 'diagnosis_report.md'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
