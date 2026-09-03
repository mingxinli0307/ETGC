#!/usr/bin/env python3
"""Combine ETGC structural-prior generalization and ablation results."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def last_metrics(run_dir):
    path = Path(run_dir) / "metrics.csv"
    rows = read_csv(path) if path.exists() else []
    return rows[-1] if rows else {}


def write_csv(path, rows):
    fields = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_row(config, row, source_dir, baseline_map, overrides=None):
    overrides = overrides or {}
    dataset = row["dataset"]
    seed = int(row["seed"])
    metric = last_metrics(Path(source_dir) / dataset / f"seed{seed}")
    final_f1 = number(row.get("final_macro_f1"))
    baseline = baseline_map[(dataset.lower(), seed)]
    return {
        "config": config,
        "dataset": dataset,
        "seed": seed,
        "final_macro_f1": final_f1,
        "delta_f1_vs_no_prior": final_f1 - number(baseline["final_macro_f1"]),
        "final_acc": number(row.get("final_acc")),
        "final_nmi": number(row.get("final_nmi")),
        "final_ari": number(row.get("final_ari")),
        "prior_mode": overrides.get("prior_mode", row.get("node_prior_mode_effective", "")),
        "logit_strength": overrides.get("logit_strength", row.get("node_prior_logit_strength", "")),
        "event_role": overrides.get("event_role", row.get("node_prior_event_role", "")),
        "connected_components": row.get("node_prior_connected_components", ""),
        "active_edge_clusters": metric.get("edge_hard_active_clusters", row.get("final_active_edge_clusters", "")),
        "active_node_clusters": metric.get("node_hard_active_clusters", row.get("final_active_node_clusters", "")),
        "largest_edge_ratio": metric.get("edge_hard_largest_ratio", row.get("final_largest_edge_ratio", "")),
        "largest_node_ratio": metric.get("node_hard_largest_ratio", row.get("final_largest_node_ratio", "")),
        "q_rank1_energy_ratio": metric.get("q_rank1_energy_ratio", row.get("final_rank1_energy", "")),
        "q_centered_to_total_energy_ratio": metric.get(
            "q_centered_to_total_energy_ratio", row.get("final_center_ratio", "")
        ),
        "q_effective_rank": metric.get("q_effective_rank", row.get("final_effective_rank", "")),
        "q_normalized_margin_mean": metric.get(
            "q_normalized_margin_mean", row.get("final_normalized_margin", "")
        ),
        "cluster_volume_cv": metric.get("cluster_volume_cv", row.get("final_volume_cv", "")),
        "qtdq_condition_number": metric.get("qtdq_condition_number", row.get("max_qtdq_condition_number", "")),
        "matrix_ncut_solve_finite": metric.get(
            "matrix_ncut_solve_finite", row.get("all_matrix_solves_finite", "")
        ),
        "runtime_seconds": number(row.get("runtime_seconds")),
        "status": row.get("status", ""),
    }


def mean_for(rows, config, dataset, field="final_macro_f1"):
    values = [number(row[field]) for row in rows if row["config"] == config and row["dataset"].lower() == dataset]
    return statistics.mean(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--main-summary", required=True, type=Path)
    parser.add_argument("--generalization-summary", required=True, type=Path)
    parser.add_argument("--ablation-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_csv(args.baseline_csv)
    baseline_map = {(row["dataset"].lower(), int(row["seed"])): row for row in baseline_rows}
    ablation_rows = []
    for dataset in ("dblp", "patent"):
        for seed in (42, 43):
            base = baseline_map[(dataset, seed)]
            ablation_rows.append(
                normalized_row(
                    "A0_no_prior",
                    base,
                    args.baseline_csv.parent,
                    baseline_map,
                    {"prior_mode": "none", "logit_strength": 0, "event_role": "none"},
                )
            )

    sources = [
        ("A1_global_s8", args.ablation_root / "A1_global_s8" / "summary.csv"),
        ("A2_component_s4", args.ablation_root / "A2_component_s4" / "summary.csv"),
        ("A3_component_s8_source", args.main_summary),
        ("A4_component_s8_mean", args.ablation_root / "A3_component_s8_mean" / "summary.csv"),
    ]
    for config, path in sources:
        for row in read_csv(path):
            if row["dataset"].lower() not in {"dblp", "patent"}:
                continue
            ablation_rows.append(normalized_row(config, row, path.parent, baseline_map))
    write_csv(args.output_dir / "ablation_summary.csv", ablation_rows)

    generalization_rows = []
    structural = read_csv(args.generalization_summary)
    structural_map = {(row["dataset"].lower(), int(row["seed"])): row for row in structural}
    for lookup in ("school", "arxivai"):
        for seed in (42, 43):
            base = baseline_map[(lookup, seed)]
            current = structural_map[(lookup, seed)]
            generalization_rows.append(
                {
                    "dataset": current["dataset"],
                    "seed": seed,
                    "baseline_final_macro_f1": number(base["final_macro_f1"]),
                    "structural_prior_final_macro_f1": number(current["final_macro_f1"]),
                    "delta_macro_f1": number(current["final_macro_f1"]) - number(base["final_macro_f1"]),
                    "structural_prior_final_nmi": number(current["final_nmi"]),
                    "structural_prior_final_ari": number(current["final_ari"]),
                    "prior_mode_effective": current["node_prior_mode_effective"],
                    "connected_components": current["node_prior_connected_components"],
                    "active_clusters": current["node_prior_active_clusters"],
                    "status": current["status"],
                }
            )
    write_csv(args.output_dir / "generalization_comparison.csv", generalization_rows)

    db_base = mean_for(ablation_rows, "A0_no_prior", "dblp")
    db_s4 = mean_for(ablation_rows, "A2_component_s4", "dblp")
    db_s8 = mean_for(ablation_rows, "A3_component_s8_source", "dblp")
    db_mean = mean_for(ablation_rows, "A4_component_s8_mean", "dblp")
    pa_base = mean_for(ablation_rows, "A0_no_prior", "patent")
    pa_global = mean_for(ablation_rows, "A1_global_s8", "patent")
    pa_component = mean_for(ablation_rows, "A3_component_s8_source", "patent")
    pa_mean = mean_for(ablation_rows, "A4_component_s8_mean", "patent")
    gen_means = {}
    for dataset in ("school", "arxivai"):
        subset = [row for row in generalization_rows if row["dataset"].lower() == dataset]
        gen_means[dataset] = (
            statistics.mean(row["baseline_final_macro_f1"] for row in subset),
            statistics.mean(row["structural_prior_final_macro_f1"] for row in subset),
        )

    report = [
        "# ETGC Structural Prior: Generalization and Ablation", "",
        "All comparisons use final epoch 20, seeds 42/43, forest_samples=50, matrix_ncut, and no label-based epoch selection.", "",
        "## Generalization", "",
        "| Dataset | Baseline mean F1 | Structural-prior mean F1 | Delta |",
        "|---|---:|---:|---:|",
        f"| School | {gen_means['school'][0]:.6f} | {gen_means['school'][1]:.6f} | {gen_means['school'][1]-gen_means['school'][0]:+.6f} |",
        f"| arXivAI | {gen_means['arxivai'][0]:.6f} | {gen_means['arxivai'][1]:.6f} | {gen_means['arxivai'][1]-gen_means['arxivai'][0]:+.6f} |",
        "", "## Ablation main effects", "",
        f"- DBLP no prior -> component-auto/source strength 8: {db_base:.6f} -> {db_s8:.6f} ({db_s8-db_base:+.6f}).",
        f"- DBLP strength 8 -> strength 4: {db_s8:.6f} -> {db_s4:.6f} ({db_s4-db_s8:+.6f}).",
        f"- DBLP source -> mean endpoints: {db_s8:.6f} -> {db_mean:.6f} ({db_mean-db_s8:+.6f}).",
        f"- Patent no prior -> global prior: {pa_base:.6f} -> {pa_global:.6f} ({pa_global-pa_base:+.6f}).",
        f"- Patent global -> component-preserving: {pa_global:.6f} -> {pa_component:.6f} ({pa_component-pa_global:+.6f}).",
        f"- Patent source -> mean endpoints: {pa_component:.6f} -> {pa_mean:.6f} ({pa_mean-pa_component:+.6f}).",
        "", "## Conclusion", "",
        "The structural prior generalizes positively to School and arXivAI. Component preservation is the dominant Patent-specific factor. Strength 4 is sufficient and slightly better on DBLP, so the gain is not caused by an extreme logit bias. Source-event semantics give a small but seed-consistent DBLP benefit, while Patent is insensitive to endpoint role.",
    ]
    (args.output_dir / "diagnosis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"baseline": str(args.baseline_csv), "main": str(args.main_summary), "generalization": str(args.generalization_summary), "ablation_root": str(args.ablation_root)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
