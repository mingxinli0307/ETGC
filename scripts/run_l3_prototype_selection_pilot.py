#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from run_l3_multiseed_validation import command, complete, cleanup, number, read_json, stages_for, stage_value


MODEL_SEEDS = [42, 44, 48]
PROTOTYPE_SEEDS = list(range(42, 50))


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def init_complete(run_dir):
    result = read_json(Path(run_dir) / "result.json")
    stages = stages_for(run_dir)
    return (
        result.get("status") == "success"
        and (Path(run_dir) / "config.json").exists()
        and bool(stages.get("after_prototype_initialization") or stages.get("after_cluster_initialization"))
    )


def run_command(args, cmd, run_dir):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (Path(run_dir) / "train.log").open("w", encoding="utf-8") as log:
        log.write("cmd=" + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=args.root_dir, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nscript_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"script_exit_code={proc.returncode}\n")
    cleanup(run_dir)
    return proc.returncode


def init_row(model_seed, prototype_seed, run_dir):
    stages = stages_for(run_dir)
    stage = stages.get("after_prototype_initialization") or stages.get("after_cluster_initialization") or {}
    config = read_json(Path(run_dir) / "config.json")
    init_info = config.get("model_init_info", {}) or {}
    return {
        "model_seed": model_seed,
        "prototype_seed": prototype_seed,
        "initial_macro_f1_diagnostic_only": stage_value(stage, "Macro_F1"),
        "initial_rank1_energy": stage_value(stage, "q_rank1_energy_ratio"),
        "initial_center_ratio": stage_value(stage, "q_centered_to_total_energy_ratio"),
        "initial_effective_rank": stage_value(stage, "q_effective_rank"),
        "initial_normalized_margin": stage_value(stage, "q_normalized_margin_mean"),
        "initial_volume_cv": stage_value(stage, "cluster_volume_cv"),
        "initial_qtdq_condition_number": stage_value(stage, "qtdq_condition_number"),
        "prototype_feature_checksum": init_info.get("prototype_feature_checksum", ""),
        "prototype_center_checksum": init_info.get("prototype_center_checksum", ""),
        "initial_q_checksum": init_info.get("initial_Q_summary_checksum", ""),
        "status": read_json(Path(run_dir) / "result.json").get("status", "missing"),
    }


def training_row(model_seed, role, prototype_seed, run_dir, selected_margin):
    result = read_json(Path(run_dir) / "result.json")
    stages = stages_for(run_dir)
    final = stages.get("final_epoch") or stages.get("epoch_20") or {}
    return {
        "model_seed": model_seed,
        "selection_role": role,
        "prototype_seed": prototype_seed,
        "selection_metric": "initial_normalized_margin",
        "selected_initial_normalized_margin": selected_margin,
        "best_macro_f1": result.get("best_metrics", {}).get("Macro_F1", ""),
        "best_epoch": result.get("best_epoch", ""),
        "final_macro_f1": result.get("final_metrics", {}).get("Macro_F1", stage_value(final, "Macro_F1")),
        "final_nmi": result.get("final_metrics", {}).get("NMI", stage_value(final, "NMI")),
        "final_ari": result.get("final_metrics", {}).get("ARI", stage_value(final, "ARI")),
        "final_rank1_energy": stage_value(final, "q_rank1_energy_ratio"),
        "final_center_ratio": stage_value(final, "q_centered_to_total_energy_ratio"),
        "final_normalized_margin": stage_value(final, "q_normalized_margin_mean"),
        "final_active_edge_clusters": stage_value(final, "num_active_edge_clusters"),
        "final_active_node_clusters": stage_value(final, "num_active_node_clusters"),
        "runtime_seconds": result.get("runtime_seconds", ""),
        "status": result.get("status", "missing"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve()
    args.output_dir = args.output_dir.resolve()
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
        "method": "ETGC",
        "purpose": "unsupervised prototype multi-start selection pilot for L3",
        "dataset": "school",
        "model_seeds": MODEL_SEEDS,
        "prototype_candidate_seeds": PROTOTYPE_SEEDS,
        "selection_rule": "maximum initial q_normalized_margin_mean; labels excluded",
        "trained_roles": ["top", "bottom"],
        "epoch": 20,
        "cluster_loss_type": "matrix_ncut",
        "orth_type": "orth",
        "fixed_c6_configuration": True,
        "save_embeddings": 0,
        "checkpoint": "disabled/not produced by edge_main.py",
    }
    (code_info / "common_config.json").write_text(json.dumps(common, indent=2, sort_keys=True), encoding="utf-8")

    init_rows = []
    failed = []
    for model_seed in MODEL_SEEDS:
        for prototype_seed in PROTOTYPE_SEEDS:
            run_dir = args.output_dir / "initializations" / f"model{model_seed}" / f"prototype{prototype_seed}"
            if args.no_resume or not init_complete(run_dir):
                cmd = command(args, model_seed, run_dir, prototype_seed=prototype_seed, epoch=0, init_only=True)
                print(f"init model={model_seed} prototype={prototype_seed}", flush=True)
                if run_command(args, cmd, run_dir) != 0:
                    failed.append(f"init_m{model_seed}_p{prototype_seed}")
            init_rows.append(init_row(model_seed, prototype_seed, run_dir))

    init_fields = list(init_rows[0].keys())
    write_csv(args.output_dir / "initialization_candidates.csv", init_fields, init_rows)
    training_rows = []
    selections = []
    for model_seed in MODEL_SEEDS:
        candidates = [row for row in init_rows if row["model_seed"] == model_seed and row["status"] == "success"]
        candidates.sort(key=lambda row: number(row["initial_normalized_margin"]))
        if len(candidates) != len(PROTOTYPE_SEEDS):
            failed.append(f"selection_m{model_seed}")
            continue
        chosen = [("bottom", candidates[0]), ("top", candidates[-1])]
        feature_checksums = {row["prototype_feature_checksum"] for row in candidates}
        selections.append({
            "model_seed": model_seed,
            "prototype_feature_checksum_fixed": len(feature_checksums) == 1 and "" not in feature_checksums,
            "bottom_prototype_seed": candidates[0]["prototype_seed"],
            "bottom_initial_normalized_margin": candidates[0]["initial_normalized_margin"],
            "top_prototype_seed": candidates[-1]["prototype_seed"],
            "top_initial_normalized_margin": candidates[-1]["initial_normalized_margin"],
        })
        for role, candidate in chosen:
            prototype_seed = int(candidate["prototype_seed"])
            run_dir = args.output_dir / "training" / f"model{model_seed}" / f"{role}_prototype{prototype_seed}"
            if args.no_resume or not complete(run_dir):
                cmd = command(args, model_seed, run_dir, prototype_seed=prototype_seed, epoch=20, init_only=False)
                print(f"train model={model_seed} role={role} prototype={prototype_seed}", flush=True)
                if run_command(args, cmd, run_dir) != 0:
                    failed.append(f"train_m{model_seed}_{role}_p{prototype_seed}")
            training_rows.append(training_row(
                model_seed, role, prototype_seed, run_dir, candidate["initial_normalized_margin"]
            ))

    write_csv(args.output_dir / "selection.csv", list(selections[0].keys()), selections)
    training_fields = list(training_rows[0].keys())
    write_csv(args.output_dir / "training_comparison.csv", training_fields, training_rows)
    deltas = []
    report = [
        "# ETGC L3 Prototype Selection Pilot", "",
        "Selection uses only the initial normalized margin. Ground-truth initialization F1 is logged for diagnosis but excluded from selection.", "",
        "| Model seed | Bottom prototype | Bottom init margin | Bottom final F1 | Top prototype | Top init margin | Top final F1 | Delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_seed in MODEL_SEEDS:
        bottom = next(row for row in training_rows if row["model_seed"] == model_seed and row["selection_role"] == "bottom")
        top = next(row for row in training_rows if row["model_seed"] == model_seed and row["selection_role"] == "top")
        delta = number(top["final_macro_f1"]) - number(bottom["final_macro_f1"])
        deltas.append(delta)
        report.append(
            f"| {model_seed} | {bottom['prototype_seed']} | {number(bottom['selected_initial_normalized_margin']):.6f} | "
            f"{number(bottom['final_macro_f1']):.6f} | {top['prototype_seed']} | "
            f"{number(top['selected_initial_normalized_margin']):.6f} | {number(top['final_macro_f1']):.6f} | {delta:.6f} |"
        )
    report += [
        "", "## Decision", "",
        f"- Top-margin final F1 exceeded bottom-margin final F1 in {sum(delta > 0 for delta in deltas)}/{len(deltas)} model seeds.",
        f"- Mean top-minus-bottom final F1 delta={sum(deltas) / len(deltas):.6f}.",
        "- Treat the margin rule as supported only if the direction is consistent across all or nearly all model seeds.",
        "- This pilot evaluates basin selection, not a new ETGC loss or model component.",
    ]
    (args.output_dir / "diagnosis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"run_failures={failed}")
    print(f"report={args.output_dir / 'diagnosis_report.md'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
