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


SEEDS = list(range(42, 50))
REUSED_SEEDS = {42, 43}
ALLOWED = {"config.json", "metrics.csv", "diagnostic.json", "result.json", "train.log"}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path):
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except Exception:
        return []


def number(value, default=math.nan):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def fmt(value):
    value = number(value)
    return "nan" if math.isnan(value) else f"{value:.6g}"


def stages_for(run_dir):
    payload = read_json(Path(run_dir) / "diagnostic.json")
    stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
    for key, value in payload.items():
        if isinstance(value, dict) and key not in stages:
            stages[key] = value
    return stages


def stage_value(stage, key, default=""):
    aliases = {"cluster_volume_cv": "cluster_volume_coefficient_of_variation"}
    if key in stage:
        return stage[key]
    alias = aliases.get(key)
    return stage.get(alias, default) if alias else default


def complete(run_dir):
    result = read_json(Path(run_dir) / "result.json")
    metrics = Path(run_dir) / "metrics.csv"
    stages = stages_for(run_dir)
    if result.get("status") != "success" or not metrics.exists() or not (Path(run_dir) / "config.json").exists():
        return False
    rows = max(0, len(metrics.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    return rows >= 20 and bool(stages.get("epoch_20")) and bool(stages.get("final_epoch"))


def cleanup(run_dir):
    for item in Path(run_dir).iterdir():
        if item.name in ALLOWED:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def copy_reference(reference_dir, seed, run_dir):
    source = Path(reference_dir) / "L3_matrix_orth" / f"seed{seed}"
    if not complete(source):
        raise RuntimeError(f"Incomplete phase-2 reference: {source}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    for name in ALLOWED:
        shutil.copy2(source / name, Path(run_dir) / name)
    cleanup(run_dir)


def command(
    args,
    seed,
    run_dir,
    prototype_seed=None,
    epoch=20,
    init_only=False,
    dataset="school",
    cluster_loss_type="matrix_ncut",
    orth_type="orth",
    forest_samples=5,
):
    if prototype_seed is None:
        prototype_seed = seed
    values = {
        "dataset": dataset, "directed": 0, "device": args.device,
        "seed": seed, "model_seed": seed, "prototype_seed": prototype_seed, "forest_seed": 20260725,
        "data_root": str(args.asset_root / "dataset"), "emb_root": str(args.asset_root / "emb"),
        "pretrain_emb_dir": str(args.asset_root / "pretrain"), "cache_dir": str(args.asset_root / "cache"),
        "batch_size": 512, "epoch": epoch, "learning_rate": 1e-4,
        "edge_dim": 128, "time_dim": 32, "edge_hidden_dim": 128, "cluster_hidden_dim": 64,
        "time_feature_mode": "history", "edge_encoder_mode": "mlp", "cluster_head_type": "legacy_mlp",
        "alpha": 0.2, "T": 4, "beta": 5.0, "edge_neighbor_k": -1, "edge_ppr_topk": -1,
        "affinity_sparsify": "symmetric_union_knn", "edge_ppr_method": "temporal_state_forest",
        "forest_samples": int(forest_samples), "ncut_scope": "global", "cluster_loss_type": cluster_loss_type,
        "orth_type": orth_type, "global_q_chunk_size": 8192, "global_ncut_row_block_size": 65536,
        "global_warmup_epochs": 0, "prox_warmup_epochs": 0, "quiet": 1,
        "lambda_prox": 0.0, "lambda_edge_ncut": 0.5, "lambda_orth": 1.0,
        "lambda_proj": 0.0, "lambda_bal": 0.0, "lambda_node_anchor": 0.0,
        "node_emb_mode": "frozen", "node_emb_lr": 1e-5, "prox_similarity_mode": "event_dot",
        "cluster_output_bias_mode": "zero", "cluster_input_norm": "layernorm",
        "cluster_init_mode": "prototype", "prototype_sample_size": 20000,
        "prototype_lloyd_iters": 10, "overnight_diagnostic": 0,
        "loss_formulation_diagnostic": 1, "diagnostic_epochs": ",".join(map(str, range(1, 21))),
        "diagnostic_stages": 0, "uniform_collapse_diagnostic": 1,
        "diagnostic_output_dir": str(run_dir), "diagnostic_only_first_epoch": 0,
        "output_dir": str(run_dir), "eval_every": 1, "save_embeddings": 0, "init_only": int(init_only),
    }
    result = [args.python_bin, "edge_main.py"]
    for key, value in values.items():
        result.extend([f"--{key}", str(value)])
    return result


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = 0.5 * (position + end - 1) + 1.0
        for offset in range(position, end):
            result[order[offset]] = rank
        position = end
    return result


def pearson(left, right):
    pairs = [(number(x), number(y)) for x, y in zip(left, right)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    x, y = zip(*pairs)
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in pairs)
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return numerator / denominator if denominator else math.nan


def spearman(left, right):
    pairs = [(number(x), number(y)) for x, y in zip(left, right)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    x, y = zip(*pairs)
    return pearson(ranks(x), ranks(y))


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(args):
    fields = [
        "seed", "source", "initial_macro_f1", "best_macro_f1", "best_epoch", "final_macro_f1",
        "best_to_final_drop", "final_nmi", "final_ari", "initial_rank1_energy", "final_rank1_energy",
        "initial_center_ratio", "final_center_ratio", "initial_effective_rank", "final_effective_rank",
        "initial_normalized_margin", "final_normalized_margin", "initial_volume_cv", "final_volume_cv",
        "initial_qtdq_condition_number", "max_qtdq_condition_number", "final_active_edge_clusters",
        "final_active_node_clusters", "all_matrix_solves_finite", "runtime_seconds", "status",
    ]
    rows = []
    for seed in SEEDS:
        run_dir = args.output_dir / f"seed{seed}"
        config = read_json(run_dir / "config.json")
        result = read_json(run_dir / "result.json")
        metrics = read_csv(run_dir / "metrics.csv")
        stages = stages_for(run_dir)
        init = stages.get("after_cluster_initialization") or stages.get("after_prototype_initialization") or {}
        final = stages.get("final_epoch") or stages.get("epoch_20") or {}
        valid = [row for row in metrics if math.isfinite(number(row.get("Macro_F1")))]
        best = max(valid, key=lambda row: number(row.get("Macro_F1")), default={})
        best_f1 = number(result.get("best_metrics", {}).get("Macro_F1", best.get("Macro_F1")))
        final_f1 = number(result.get("final_metrics", {}).get("Macro_F1", stage_value(final, "Macro_F1")))
        conditions = [number(row.get("qtdq_condition_number")) for row in valid]
        conditions = [value for value in conditions if math.isfinite(value)]
        finite_flags = [str(row.get("matrix_ncut_solve_finite", "")).lower() for row in valid]
        row = {
            "seed": seed, "source": "reused" if seed in REUSED_SEEDS else "new",
            "initial_macro_f1": stage_value(init, "Macro_F1"), "best_macro_f1": best_f1,
            "best_epoch": result.get("best_epoch", best.get("epoch", "")), "final_macro_f1": final_f1,
            "best_to_final_drop": best_f1 - final_f1, "final_nmi": stage_value(final, "NMI"),
            "final_ari": stage_value(final, "ARI"), "initial_rank1_energy": stage_value(init, "q_rank1_energy_ratio"),
            "final_rank1_energy": stage_value(final, "q_rank1_energy_ratio"),
            "initial_center_ratio": stage_value(init, "q_centered_to_total_energy_ratio"),
            "final_center_ratio": stage_value(final, "q_centered_to_total_energy_ratio"),
            "initial_effective_rank": stage_value(init, "q_effective_rank"),
            "final_effective_rank": stage_value(final, "q_effective_rank"),
            "initial_normalized_margin": stage_value(init, "q_normalized_margin_mean"),
            "final_normalized_margin": stage_value(final, "q_normalized_margin_mean"),
            "initial_volume_cv": stage_value(init, "cluster_volume_cv"),
            "final_volume_cv": stage_value(final, "cluster_volume_cv"),
            "initial_qtdq_condition_number": stage_value(init, "qtdq_condition_number"),
            "max_qtdq_condition_number": max(conditions) if conditions else "",
            "final_active_edge_clusters": stage_value(final, "num_active_edge_clusters"),
            "final_active_node_clusters": stage_value(final, "num_active_node_clusters"),
            "all_matrix_solves_finite": bool(finite_flags) and all(flag == "true" for flag in finite_flags),
            "runtime_seconds": result.get("runtime_seconds", ""), "status": result.get("status", "missing"),
        }
        if int(config.get("model_seed", -1)) != seed or int(config.get("prototype_seed", -1)) != seed:
            row["status"] = "seed_mismatch"
        rows.append(row)
    write_csv(args.output_dir / "multiseed_comparison.csv", fields, rows)

    final_values = [number(row["final_macro_f1"]) for row in rows]
    best_values = [number(row["best_macro_f1"]) for row in rows]
    predictors = [
        ("initial_rank1_energy", "Initial Rank1"),
        ("initial_center_ratio", "Initial center ratio"),
        ("initial_effective_rank", "Initial effective rank"),
        ("initial_normalized_margin", "Initial normalized margin"),
        ("initial_volume_cv", "Initial volume CV"),
        ("initial_qtdq_condition_number", "Initial QTDQ condition"),
    ]
    correlations = []
    for key, label in predictors:
        values = [row[key] for row in rows]
        correlations.append((label, pearson(values, final_values), spearman(values, final_values)))
    report = [
        "# ETGC L3 Multi-Seed Stability Validation", "",
        "Scope: School, seeds 42-49, matrix_ncut + orth, fixed C6 configuration, 20 epochs.", "",
        "## Results", "",
        "| Seed | Source | Initial F1 | Best F1 | Best epoch | Final F1 | Drop | Final NMI | Final ARI | Final Rank1 | Center ratio | Norm margin | QTDQ max |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['seed']} | {row['source']} | {fmt(row['initial_macro_f1'])} | {fmt(row['best_macro_f1'])} | "
            f"{row['best_epoch']} | {fmt(row['final_macro_f1'])} | {fmt(row['best_to_final_drop'])} | "
            f"{fmt(row['final_nmi'])} | {fmt(row['final_ari'])} | {fmt(row['final_rank1_energy'])} | "
            f"{fmt(row['final_center_ratio'])} | {fmt(row['final_normalized_margin'])} | "
            f"{fmt(row['max_qtdq_condition_number'])} |"
        )
    report += [
        "", "## Aggregate Stability", "",
        f"- Final F1 mean={fmt(statistics.mean(final_values))}, std={fmt(statistics.stdev(final_values))}, "
        f"min={fmt(min(final_values))}, max={fmt(max(final_values))}.",
        f"- Best F1 mean={fmt(statistics.mean(best_values))}, std={fmt(statistics.stdev(best_values))}.",
        f"- Seeds with final F1 >= 0.90: {sum(value >= 0.90 for value in final_values)}/{len(final_values)}.",
        f"- Seeds with final F1 >= 0.80: {sum(value >= 0.80 for value in final_values)}/{len(final_values)}.",
        f"- All runs retained all 9 edge/node clusters: "
        f"{all(int(float(row['final_active_edge_clusters'])) == 9 and int(float(row['final_active_node_clusters'])) == 9 for row in rows)}.",
        f"- All matrix solves finite: {all(row['all_matrix_solves_finite'] for row in rows)}.",
        "", "## Exploratory Unsupervised Predictors", "",
        "These correlations use only eight seeds and are diagnostic, not selection-rule validation.", "",
        "| Initial diagnostic | Pearson with final F1 | Spearman with final F1 |", "|---|---:|---:|",
    ]
    for label, pearson_value, spearman_value in correlations:
        report.append(f"| {label} | {fmt(pearson_value)} | {fmt(spearman_value)} |")
    report += [
        "", "## Decision", "",
        "Do not formalize L3 solely from a high single seed if final-F1 variance remains large. "
        "Use the distribution, drift, and unsupervised-diagnostic correlations to choose the next focused intervention.",
    ]
    (args.output_dir / "diagnosis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.root_dir = args.root_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.asset_root = args.asset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_info = args.output_dir / "code_info"
    code_info.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.root_dir, capture_output=True, text=True).stdout.strip()
    log_line = subprocess.run(["git", "log", "-1", "--oneline"], cwd=args.root_dir, capture_output=True, text=True).stdout.strip()
    (code_info / "commit.txt").write_text(commit + "\n" + log_line + "\n", encoding="utf-8")
    environment = [f"python={args.python_bin}", f"device={args.device}"]
    environment += [f"{key}={os.environ.get(key, '')}" for key in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]]
    (code_info / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    common = {
        "method": "ETGC", "purpose": "L3 multi-seed stability validation", "dataset": "school",
        "seeds": SEEDS, "reused_seeds": sorted(REUSED_SEEDS), "new_seeds": sorted(set(SEEDS) - REUSED_SEEDS),
        "epoch": 20, "cluster_loss_type": "matrix_ncut", "orth_type": "orth",
        "model_seed_equals_prototype_seed": True, "phase2_reference_dir": str(args.reference_dir),
        "fixed_c6_configuration": True, "asset_root": str(args.asset_root), "save_embeddings": 0,
        "checkpoint": "disabled/not produced by edge_main.py",
    }
    (code_info / "common_config.json").write_text(json.dumps(common, indent=2, sort_keys=True), encoding="utf-8")

    failed = []
    for seed in SEEDS:
        run_dir = args.output_dir / f"seed{seed}"
        if seed in REUSED_SEEDS:
            print(f"reuse seed={seed}", flush=True)
            try:
                copy_reference(args.reference_dir, seed, run_dir)
            except Exception as exc:
                print(f"reuse_failed seed={seed} error={exc}", flush=True)
                failed.append(seed)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_resume and complete(run_dir):
            print(f"skip seed={seed}", flush=True)
            cleanup(run_dir)
            continue
        cmd = command(args, seed, run_dir)
        print(f"run seed={seed}", flush=True)
        started = time.time()
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            log.write("cmd=" + " ".join(cmd) + "\n")
            proc = subprocess.run(cmd, cwd=args.root_dir, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"\nscript_runtime_seconds={time.time() - started:.6f}\n")
            log.write(f"script_exit_code={proc.returncode}\n")
        if proc.returncode != 0:
            failed.append(seed)
        cleanup(run_dir)
    summarize(args)
    print(f"run_failures={failed}")
    print(f"comparison={args.output_dir / 'multiseed_comparison.csv'}")
    print(f"report={args.output_dir / 'diagnosis_report.md'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
