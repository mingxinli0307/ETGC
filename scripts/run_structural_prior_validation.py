#!/usr/bin/env python3
"""Run the label-free component structural prior with the fixed ETGC objective."""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from run_l3_multiseed_validation import cleanup, command, complete, read_json


def parse_ints(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_one(args, dataset, seed, run_dir):
    cmd = command(
        args,
        seed,
        run_dir,
        prototype_seed=seed,
        epoch=args.epochs,
        init_only=args.init_only,
        dataset=dataset,
        cluster_loss_type="matrix_ncut",
        orth_type="orth",
        forest_samples=50,
        cluster_head_type="legacy_mlp",
    )
    cmd.extend(
        [
            "--node_prior_mode", "component_structural",
            "--node_prior_seed", str(seed),
            "--node_prior_restarts", str(args.kmeans_restarts),
            "--node_prior_bisecting_restarts", str(args.bisecting_restarts),
            "--node_prior_logit_strength", str(args.logit_strength),
            "--lambda_node_prior", "0.0",
        ]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("cmd=" + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=args.root_dir, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nscript_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"script_exit_code={proc.returncode}\n")
    cleanup(run_dir)
    return proc.returncode


def summarize(output_dir, datasets, seeds):
    fields = [
        "dataset", "seed", "status", "final_macro_f1", "final_acc", "final_nmi",
        "final_ari", "runtime_seconds", "node_prior_mode_effective",
        "node_prior_connected_components", "node_prior_active_clusters",
        "node_prior_logit_strength", "cluster_loss_type", "forest_samples",
    ]
    rows = []
    for dataset in datasets:
        for seed in seeds:
            result = read_json(output_dir / dataset / f"seed{seed}" / "result.json")
            final = result.get("final_metrics", {})
            prior = result.get("node_prior_info", {})
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "status": result.get("status", "missing"),
                    "final_macro_f1": final.get("Macro_F1", ""),
                    "final_acc": final.get("ACC", ""),
                    "final_nmi": final.get("NMI", ""),
                    "final_ari": final.get("ARI", ""),
                    "runtime_seconds": result.get("runtime_seconds", ""),
                    "node_prior_mode_effective": prior.get("node_prior_mode_effective", ""),
                    "node_prior_connected_components": prior.get("node_prior_connected_components", ""),
                    "node_prior_active_clusters": prior.get("node_prior_active_clusters", ""),
                    "node_prior_logit_strength": result.get("node_prior_logit_strength", ""),
                    "cluster_loss_type": result.get("cluster_loss_type", ""),
                    "forest_samples": result.get("forest_samples", ""),
                }
            )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", default="dblp,patent")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-only", type=int, choices=(0, 1), default=0)
    parser.add_argument("--logit-strength", type=float, default=8.0)
    parser.add_argument("--kmeans-restarts", type=int, default=500)
    parser.add_argument("--bisecting-restarts", type=int, default=50)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve()
    args.asset_root = args.asset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = parse_ints(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_info = args.output_dir / "code_info"
    code_info.mkdir(exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.root_dir, text=True, capture_output=True, check=True
    ).stdout.strip()
    (code_info / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    common = {
        "datasets": datasets,
        "seeds": seeds,
        "epochs": args.epochs,
        "init_only": bool(args.init_only),
        "forest_samples": 50,
        "cluster_loss_type": "matrix_ncut",
        "orth_type": "orth",
        "cluster_head_type": "legacy_mlp",
        "cluster_output_bias_mode": "zero",
        "cluster_input_norm": "layernorm",
        "cluster_init_mode": "prototype",
        "node_prior_mode": "component_structural",
        "node_prior_logit_strength": args.logit_strength,
        "node_prior_kmeans_restarts": args.kmeans_restarts,
        "node_prior_bisecting_restarts": args.bisecting_restarts,
        "label_used_for_training_or_selection": False,
        "python": args.python_bin,
        "device": args.device,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (code_info / "common_config.json").write_text(
        json.dumps(common, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    failed = []
    for dataset in datasets:
        for seed in seeds:
            run_dir = args.output_dir / dataset / f"seed{seed}"
            if not args.no_resume and complete(run_dir):
                print(f"skip dataset={dataset} seed={seed}", flush=True)
                continue
            print(f"run dataset={dataset} seed={seed}", flush=True)
            if run_one(args, dataset, seed, run_dir) != 0:
                failed.append((dataset, seed))
    rows = summarize(args.output_dir, datasets, seeds)
    print(json.dumps(rows, indent=2), flush=True)
    if failed:
        raise SystemExit(f"failed runs: {failed}")


if __name__ == "__main__":
    main()
