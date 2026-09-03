#!/usr/bin/env python3
"""Strict paired ETGC objective validation from reusable initialization snapshots."""

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


DATASETS = ("dblp", "patent")
SEEDS = (42, 43, 44, 45, 46)
BRANCHES = {
    "P0_init_only": {"init_only": 1, "cut": 0.0, "orth": 0.0, "training_prior_strength": None},
    "P1_matrix_only": {"init_only": 0, "cut": 1.0, "orth": 0.0, "training_prior_strength": None},
    "P2_orth_only": {"init_only": 0, "cut": 0.0, "orth": 1.0, "training_prior_strength": None},
    "P3_matrix_orth_prior_off": {"init_only": 0, "cut": 1.0, "orth": 1.0, "training_prior_strength": 0.0},
    "P4_current_full": {"init_only": 0, "cut": 1.0, "orth": 1.0, "training_prior_strength": 4.0},
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
class Pair:
    dataset: str
    seed: int


def make_pairs() -> list[Pair]:
    return [Pair(dataset, seed) for dataset in DATASETS for seed in SEEDS]


def run_dir_for(output_dir: Path, branch: str, pair: Pair) -> Path:
    return output_dir / branch / pair.dataset / f"seed{pair.seed}"


def snapshot_for(output_dir: Path, pair: Pair) -> Path:
    return output_dir / "code_info" / "initialization_states" / f"{pair.dataset}_seed{pair.seed}.pt"


def build_command(args, pair: Pair, branch: str, run_dir: Path, snapshot: Path) -> list[str]:
    cfg = BRANCHES[branch]
    worker = SimpleNamespace(device=args.device, asset_root=args.asset_root, python_bin=args.python_bin)
    cmd = command(
        worker,
        pair.seed,
        run_dir,
        prototype_seed=pair.seed,
        epoch=args.epochs,
        init_only=bool(cfg["init_only"]),
        dataset=pair.dataset,
        cluster_loss_type="matrix_ncut",
        orth_type="orth",
        forest_samples=50,
        cluster_head_type="legacy_mlp",
    )
    for name, value in BASE_OVERRIDES.items():
        set_flag(cmd, name, value)
    set_flag(cmd, "node_prior_seed", pair.seed)
    set_flag(cmd, "global_cut_scale", cfg["cut"])
    set_flag(cmd, "global_orth_scale", cfg["orth"])
    if branch == "P0_init_only":
        set_flag(cmd, "initialization_state_out", snapshot)
    else:
        set_flag(cmd, "initialization_state_in", snapshot)
    if cfg["training_prior_strength"] is not None:
        set_flag(cmd, "node_prior_training_logit_strength", cfg["training_prior_strength"])
    return cmd


def complete(run_dir: Path, branch: str, epochs: int) -> bool:
    result = read_json(run_dir / "result.json")
    config = read_json(run_dir / "config.json")
    metrics = run_dir / "metrics.csv"
    if result.get("status") != "success" or not config or not metrics.exists():
        return False
    rows = max(0, len(metrics.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    return rows >= (1 if BRANCHES[branch]["init_only"] else epochs)


def run_one(args, pair: Pair, branch: str, snapshot: Path) -> int:
    run_dir = run_dir_for(args.output_dir, branch, pair)
    can_resume = complete(run_dir, branch, args.epochs)
    if branch == "P0_init_only" and not snapshot.exists():
        can_resume = False
    if not args.no_resume and can_resume:
        cleanup(run_dir)
        return 0
    if branch != "P0_init_only" and not snapshot.exists():
        raise FileNotFoundError(f"Missing paired initialization snapshot: {snapshot}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, pair, branch, run_dir, snapshot)
    started = time.time()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("command=" + " ".join(map(str, cmd)) + "\n")
        process = subprocess.run(cmd, cwd=args.root_dir, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nrunner_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"runner_exit_code={process.returncode}\n")
    cleanup(run_dir)
    return int(process.returncode)


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def result_row(output_dir: Path, branch: str, pair: Pair) -> dict:
    run_dir = run_dir_for(output_dir, branch, pair)
    result = read_json(run_dir / "result.json")
    config = read_json(run_dir / "config.json")
    metrics = result.get("final_metrics", {})
    best = result.get("best_metrics", {})
    change = result.get("training_change_metrics", {})
    info = result.get("model_init_info", config.get("model_init_info", {})) or {}
    stages = stages_for(run_dir)
    final_stage = stages.get("final_epoch") or stages.get(f"epoch_{config.get('epoch', '')}") or {}
    return {
        "branch": branch,
        "dataset": pair.dataset,
        "seed": pair.seed,
        "status": result.get("status", "missing"),
        "initial_ACC": change.get("ACC_init", metrics.get("ACC", "")),
        "final_ACC": metrics.get("ACC", ""),
        "initial_Macro_F1": change.get("MacroF1_init", metrics.get("Macro_F1", "")),
        "best_Macro_F1": best.get("Macro_F1", ""),
        "final_Macro_F1": metrics.get("Macro_F1", ""),
        "initial_NMI": change.get("NMI_init", metrics.get("NMI", "")),
        "final_NMI": metrics.get("NMI", ""),
        "initial_ARI": change.get("ARI_init", metrics.get("ARI", "")),
        "final_ARI": metrics.get("ARI", ""),
        "node_prediction_ari_init_final": change.get("node_prediction_ari_init_final", 1.0 if branch == "P0_init_only" else ""),
        "edge_prediction_ari_init_final": change.get("edge_prediction_ari_init_final", 1.0 if branch == "P0_init_only" else ""),
        "q_drift_fro_normalized": change.get("q_drift_fro_normalized", 0.0 if branch == "P0_init_only" else ""),
        "q_centered_to_total_energy_ratio": final_stage.get("q_centered_to_total_energy_ratio", ""),
        "q_normalized_margin_mean": final_stage.get("q_normalized_margin_mean", ""),
        "initial_q_checksum": info.get("initial_Q_summary_checksum", ""),
        "model_state_checksum": info.get("model_state_checksum", ""),
        "Pi_cut_checksum": info.get("Pi_cut_checksum", ""),
        "node_prior_initial_strength": config.get("node_prior_logit_strength_during_initialization", 4.0),
        "node_prior_training_strength": config.get("node_prior_logit_strength", ""),
        "runtime_seconds": result.get("runtime_seconds", ""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_dir: Path, pairs: list[Pair]) -> None:
    rows = [result_row(output_dir, branch, pair) for pair in pairs for branch in BRANCHES]
    write_csv(output_dir / "paired_summary.csv", rows)
    aggregates = []
    for dataset in DATASETS:
        for branch in BRANCHES:
            group = [row for row in rows if row["dataset"] == dataset and row["branch"] == branch and row["status"] == "success"]
            item = {"dataset": dataset, "branch": branch, "successful_seeds": len(group)}
            for key in ("final_ACC", "final_Macro_F1", "final_NMI", "final_ARI", "node_prediction_ari_init_final", "edge_prediction_ari_init_final", "q_drift_fro_normalized"):
                values = [number(row.get(key)) for row in group]
                values = [value for value in values if value is not None]
                item[f"{key}_mean"] = statistics.mean(values) if values else ""
                item[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "")
            aggregates.append(item)
    write_csv(output_dir / "paired_aggregate.csv", aggregates)
    checksum_keys = ("initial_q_checksum", "model_state_checksum", "Pi_cut_checksum")
    audits = []
    for pair in pairs:
        group = [row for row in rows if row["dataset"] == pair.dataset and row["seed"] == pair.seed]
        audit = {"dataset": pair.dataset, "seed": pair.seed}
        for key in checksum_keys:
            values = {row[key] for row in group if row["status"] == "success" and row[key]}
            audit[f"{key}_matched"] = len(values) == 1 and len(group) == len(BRANCHES)
            audit[f"{key}_value"] = next(iter(values)) if len(values) == 1 else ""
        audits.append(audit)
    write_csv(output_dir / "pairing_audit.csv", audits)
    passed = all(all(row[f"{key}_matched"] for key in checksum_keys) for row in audits)
    lines = [
        "# ETGC Strict Paired Objective Validation", "",
        f"Initialization checksum audit: {'PASS' if passed else 'FAIL'}.", "",
        "P1/P2 isolate Matrix Ncut and Orth under persistent component prior. P3 removes the prior after initialization; P4 is the current persistent-prior objective.", "",
        "| Dataset | Branch | ACC | Macro-F1 | NMI | ARI | Node ARI init-final |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        values = [row.get(f"{key}_mean") for key in ("final_ACC", "final_Macro_F1", "final_NMI", "final_ARI", "node_prediction_ari_init_final")]
        lines.append("| " + " | ".join([row["dataset"], row["branch"]] + ["" if value == "" else f"{float(value):.6f}" for value in values]) + " |")
    (output_dir / "diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_code_info(args, pairs: list[Pair]) -> None:
    code = args.output_dir / "code_info"
    code.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.root_dir, text=True, capture_output=True, check=True).stdout.strip()
    (code / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    (code / "environment.json").write_text(json.dumps({"python": args.python_bin, "platform": platform.platform(), "device": args.device}, indent=2) + "\n", encoding="utf-8")
    (code / "common_config.json").write_text(json.dumps({"datasets": DATASETS, "seeds": SEEDS, "epochs": args.epochs, "branches": BRANCHES, "base_overrides": BASE_OVERRIDES, "forest_samples": 50, "cluster_loss_type": "matrix_ncut", "orth_type": "orth", "commit": commit}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--require-device-name-contains", default="A100")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-snapshots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve(); args.asset_root = args.asset_root.resolve(); args.output_dir = args.output_dir.resolve()
    validate_devices([args.device], visible_cuda_device_names(args.python_bin), args.require_device_name_contains)
    pairs = make_pairs(); args.output_dir.mkdir(parents=True, exist_ok=True); write_code_info(args, pairs)
    failures = []
    for pair in pairs:
        snapshot = snapshot_for(args.output_dir, pair)
        for branch in BRANCHES:
            cmd = build_command(args, pair, branch, run_dir_for(args.output_dir, branch, pair), snapshot)
            if args.dry_run:
                print(" ".join(map(str, cmd))); continue
            print(f"run branch={branch} dataset={pair.dataset} seed={pair.seed}", flush=True)
            code = run_one(args, pair, branch, snapshot)
            if code != 0:
                failures.append((branch, pair.dataset, pair.seed, code))
            summarize(args.output_dir, pairs)
        if not failures and not args.keep_snapshots:
            snapshot.unlink(missing_ok=True)
    if not args.dry_run:
        summarize(args.output_dir, pairs)
    print(f"pair_count={len(pairs)} branch_count={len(BRANCHES)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
