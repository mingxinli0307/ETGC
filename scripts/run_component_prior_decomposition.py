#!/usr/bin/env python3
"""Decompose raw KMeans, component-aware prior, Q projection, and prototype initialization."""

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

from run_initial_node_kmeans_baseline import run_one as run_raw_node_kmeans
from run_l3_multiseed_validation import cleanup, command, read_json, stages_for
from run_paper_objective_validation import set_flag, validate_devices, visible_cuda_device_names


DATASETS = ("school", "dblp", "patent", "arXivAI")
SEEDS = (42, 43, 44, 45, 46)
CONFIGS = (
    "B0_raw_node2vec_kmeans",
    "B1_component_prior_direct",
    "B2_component_prior_q_random",
    "B3_component_prior_q_prototype",
)
BASE_OVERRIDES = {
    "require_pretrained_node2vec": 1,
    "node_prior_mode": "component_structural",
    "node_prior_restarts": 500,
    "node_prior_bisecting_restarts": 50,
    "node_prior_logit_strength": 4.0,
    "node_prior_event_role": "source",
    "lambda_node_prior": 0.0,
    "global_cut_scale": 0.0,
    "global_orth_scale": 0.0,
}


@dataclass(frozen=True)
class Case:
    dataset: str
    seed: int


def make_cases() -> list[Case]:
    return [Case(dataset, seed) for dataset in DATASETS for seed in SEEDS]


def run_dir_for(output_dir: Path, config: str, case: Case) -> Path:
    suffix = f"seed_{case.seed}" if config == "B0_raw_node2vec_kmeans" else f"seed{case.seed}"
    return output_dir / config / case.dataset / suffix


def snapshot_for(output_dir: Path, case: Case) -> Path:
    return output_dir / "code_info" / "initialization_states" / f"{case.dataset}_seed{case.seed}.pt"


def build_edge_command(args, config: str, case: Case, run_dir: Path, snapshot: Path) -> list[str]:
    worker = SimpleNamespace(device=args.device, asset_root=args.asset_root, python_bin=args.python_bin)
    cmd = command(
        worker,
        case.seed,
        run_dir,
        prototype_seed=case.seed,
        epoch=1,
        init_only=config != "B1_component_prior_direct",
        dataset=case.dataset,
        cluster_loss_type="matrix_ncut",
        orth_type="orth",
        forest_samples=50,
        cluster_head_type="legacy_mlp",
    )
    for name, value in BASE_OVERRIDES.items():
        set_flag(cmd, name, value)
    set_flag(cmd, "node_prior_seed", case.seed)
    set_flag(cmd, "cluster_init_mode", "prototype" if config == "B3_component_prior_q_prototype" else "random")
    if config == "B2_component_prior_q_random":
        set_flag(cmd, "initialization_state_out", snapshot)
    else:
        set_flag(cmd, "initialization_state_in", snapshot)
    if config == "B1_component_prior_direct":
        set_flag(cmd, "direct_node_prior_eval", 1)
        set_flag(cmd, "init_only", 0)
    if config == "B3_component_prior_q_prototype":
        set_flag(cmd, "apply_cluster_initialization_after_state_load", 1)
    set_flag(cmd, "diagnostic_epochs", "")
    return cmd


def complete(run_dir: Path) -> bool:
    result = read_json(run_dir / "result.json")
    metrics = run_dir / "metrics.csv"
    return result.get("status") == "success" and metrics.exists() and len(metrics.read_text(encoding="utf-8", errors="ignore").splitlines()) >= 2


def raw_config(args, case: Case) -> dict:
    path = args.asset_root / "pretrain" / f"{case.dataset}_feature.emb"
    return {
        "method": "initial_node_kmeans",
        "dataset": case.dataset,
        "seed": case.seed,
        "node_count": 0,
        "event_count": 0,
        "class_count": 0,
        "node2vec_path": str(path),
        "fallback_dim": 128,
        "node_kmeans_sample_size": -1,
        "node_kmeans_lloyd_iters": 10,
        "assign_chunk_size": 8192,
        "device": args.device,
        "require_pretrained_node2vec": 1,
    }


def run_edge(args, config: str, case: Case, snapshot: Path) -> int:
    run_dir = run_dir_for(args.output_dir, config, case)
    resumable = complete(run_dir)
    if config == "B2_component_prior_q_random" and not snapshot.exists():
        resumable = False
    if not args.no_resume and resumable:
        cleanup(run_dir)
        return 0
    if config != "B2_component_prior_q_random" and not snapshot.exists():
        raise FileNotFoundError(f"Missing shared component-prior snapshot: {snapshot}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_edge_command(args, config, case, run_dir, snapshot)
    started = time.time()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("command=" + " ".join(map(str, cmd)) + "\n")
        process = subprocess.run(cmd, cwd=args.root_dir, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nrunner_runtime_seconds={time.time() - started:.6f}\nrunner_exit_code={process.returncode}\n")
    cleanup(run_dir)
    return int(process.returncode)


def result_row(output_dir: Path, config: str, case: Case) -> dict:
    run_dir = run_dir_for(output_dir, config, case)
    result = read_json(run_dir / "result.json")
    metrics = result.get("final_metrics") or result.get("metrics") or {}
    config_json = read_json(run_dir / "config.json")
    prior = result.get("node_prior_info", config_json.get("node_prior_info", {})) or {}
    stages = stages_for(run_dir)
    init = stages.get("after_cluster_initialization") or {}
    return {
        "config": config,
        "dataset": case.dataset,
        "seed": case.seed,
        "status": result.get("status", "missing"),
        "ACC": metrics.get("ACC", ""),
        "Macro_F1": metrics.get("Macro_F1", ""),
        "NMI": metrics.get("NMI", ""),
        "ARI": metrics.get("ARI", ""),
        "node_hard_active_clusters": metrics.get("node_hard_active_clusters", init.get("num_active_node_clusters", "")),
        "node_hard_largest_ratio": metrics.get("node_hard_largest_ratio", init.get("largest_node_cluster_ratio", "")),
        "edge_hard_active_clusters": metrics.get("edge_hard_active_clusters", init.get("num_active_edge_clusters", "")),
        "q_rank1_energy_ratio": init.get("q_rank1_energy_ratio", ""),
        "q_centered_to_total_energy_ratio": init.get("q_centered_to_total_energy_ratio", ""),
        "q_normalized_margin_mean": init.get("q_normalized_margin_mean", ""),
        "node_prior_mode_effective": prior.get("node_prior_mode_effective", ""),
        "uses_event_assignment_Q": metrics.get("uses_event_assignment_Q", config not in {"B0_raw_node2vec_kmeans", "B1_component_prior_direct"}),
        "uses_incidence_projection": metrics.get("uses_incidence_projection", config in {"B2_component_prior_q_random", "B3_component_prior_q_prototype"}),
        "runtime_seconds": result.get("runtime_seconds", ""),
    }


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(output_dir: Path, cases: list[Case]) -> None:
    rows = [result_row(output_dir, config, case) for case in cases for config in CONFIGS]
    write_csv(output_dir / "decomposition_summary.csv", rows)
    aggregates = []
    for dataset in DATASETS:
        for config in CONFIGS:
            group = [row for row in rows if row["dataset"] == dataset and row["config"] == config and row["status"] == "success"]
            item = {"dataset": dataset, "config": config, "successful_seeds": len(group)}
            for key in ("ACC", "Macro_F1", "NMI", "ARI", "node_hard_largest_ratio", "q_centered_to_total_energy_ratio", "q_normalized_margin_mean", "runtime_seconds"):
                values = [number(row.get(key)) for row in group]; values = [value for value in values if value is not None]
                item[f"{key}_mean"] = statistics.mean(values) if values else ""
                item[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "")
            aggregates.append(item)
    write_csv(output_dir / "decomposition_aggregate.csv", aggregates)
    lines = ["# ETGC Component-Aware Prior Decomposition", "", "All metrics are evaluated before training.", "", "| Dataset | Config | ACC | Macro-F1 | NMI | ARI |", "|---|---|---:|---:|---:|---:|"]
    for row in aggregates:
        values = [row.get(f"{key}_mean") for key in ("ACC", "Macro_F1", "NMI", "ARI")]
        lines.append("| " + " | ".join([row["dataset"], row["config"]] + ["" if value == "" else f"{float(value):.6f}" for value in values]) + " |")
    lines.extend(["", "B0 is raw Node2Vec KMeans; B1 directly evaluates component-aware node labels; B2 adds event Q and incidence projection with a random head; B3 additionally applies prototype initialization."])
    (output_dir / "diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_code_info(args) -> None:
    code = args.output_dir / "code_info"; code.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.root_dir, text=True, capture_output=True, check=True).stdout.strip()
    (code / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    (code / "environment.json").write_text(json.dumps({"python": args.python_bin, "platform": platform.platform(), "device": args.device}, indent=2) + "\n", encoding="utf-8")
    (code / "common_config.json").write_text(json.dumps({"datasets": DATASETS, "seeds": SEEDS, "configs": CONFIGS, "base_overrides": BASE_OVERRIDES, "forest_samples": 50, "commit": commit}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-device-name-contains", default="A100")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-snapshots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); args.root_dir = args.root_dir.resolve(); args.asset_root = args.asset_root.resolve(); args.output_dir = args.output_dir.resolve()
    validate_devices([args.device], visible_cuda_device_names(args.python_bin), args.require_device_name_contains)
    cases = make_cases(); args.output_dir.mkdir(parents=True, exist_ok=True); write_code_info(args)
    failures = []
    for case in cases:
        if args.dry_run:
            snapshot = snapshot_for(args.output_dir, case)
            for config in ("B2_component_prior_q_random", "B1_component_prior_direct", "B3_component_prior_q_prototype"):
                print(" ".join(map(str, build_edge_command(args, config, case, run_dir_for(args.output_dir, config, case), snapshot))))
            continue
        raw_dir = args.output_dir / "B0_raw_node2vec_kmeans"
        if args.no_resume or not complete(run_dir_for(args.output_dir, "B0_raw_node2vec_kmeans", case)):
            result = run_raw_node_kmeans(args.asset_root, raw_dir, raw_config(args, case), resume=not args.no_resume)
            if result.get("status") != "success": failures.append(("B0", case.dataset, case.seed))
        snapshot = snapshot_for(args.output_dir, case)
        for config in ("B2_component_prior_q_random", "B1_component_prior_direct", "B3_component_prior_q_prototype"):
            print(f"run config={config} dataset={case.dataset} seed={case.seed}", flush=True)
            code = run_edge(args, config, case, snapshot)
            if code != 0: failures.append((config, case.dataset, case.seed))
            summarize(args.output_dir, cases)
        if not failures and not args.keep_snapshots: snapshot.unlink(missing_ok=True)
    if not args.dry_run: summarize(args.output_dir, cases)
    print(f"case_count={len(cases)} config_count={len(CONFIGS)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
