#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


CONFIGS = [
    ("L1_trace_orth", "legacy_trace_ratio", "orth"),
    ("L2_trace_orthqa", "legacy_trace_ratio", "orthqa"),
    ("L3_matrix_orth", "matrix_ncut", "orth"),
    ("L4_matrix_orthqa", "matrix_ncut", "orthqa"),
]
SEEDS = [42, 43]


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as reader:
        return list(csv.DictReader(reader))


def finite(value, default=math.nan):
    try:
        if value in (None, "", "None", "null"):
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def number(value, default=math.nan):
    try:
        if value in (None, "", "None", "null"):
            return default
        value = float(value)
        return default if math.isnan(value) else value
    except Exception:
        return default


def fmt(value):
    value = number(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def stages_for(run_dir):
    payload = read_json(Path(run_dir) / "diagnostic.json")
    stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
    for key, value in payload.items():
        if isinstance(value, dict) and key not in stages:
            stages[key] = value
    return stages


def initial_stage(stages):
    return (
        stages.get("after_cluster_initialization")
        or stages.get("after_prototype_initialization")
        or stages.get("after_model_initialization")
        or {}
    )


def stage_value(stage, key, default=""):
    aliases = {
        "cluster_volume_cv": "cluster_volume_coefficient_of_variation",
        "q_entropy": "q_entropy_mean",
    }
    if key in stage:
        return stage[key]
    alias = aliases.get(key)
    return stage.get(alias, default) if alias else default


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow({key: row.get(key, "") for key in fieldnames})


def mean(rows, key):
    values = [finite(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--c6-reference-dir", default="")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    reference_dir = Path(args.c6_reference_dir).resolve() if args.c6_reference_dir else None

    combined_fields = [
        "loss_config", "seed", "epoch", "cluster_loss_type", "orth_type",
        "cut_loss", "orth_loss", "orth_original_loss", "orthqa_loss",
        "orth_original_value", "orthqa_value", "total_loss",
        "q_rank1_energy_ratio", "q_second_energy_ratio", "q_effective_rank", "q_numerical_rank",
        "q_centered_energy", "q_centered_effective_rank", "q_centered_numerical_rank",
        "q_centered_to_total_energy_ratio", "q_margin_mean", "q_margin_p99",
        "q_normalized_margin_mean", "q_normalized_margin_median",
        "q_normalized_margin_p90", "q_normalized_margin_p99",
        "q_entropy", "q_entropy_gap", "q_uniform_l2_mean",
        "num_active_edge_clusters", "largest_edge_cluster_ratio",
        "num_active_node_clusters", "largest_node_cluster_ratio",
        "cluster_volume_cv", "cluster_volume_max_ratio", "cluster_volume_min_ratio",
        "cluster_volume_entropy", "qtdq_min_eigenvalue", "qtdq_max_eigenvalue",
        "qtdq_condition_number", "matrix_ncut_solve_finite", "ACC", "NMI", "ARI", "Macro_F1",
    ]
    comparison_fields = [
        "config", "seed", "initial_macro_f1", "best_macro_f1", "best_epoch", "final_macro_f1",
        "best_nmi", "final_nmi", "best_ari", "final_ari",
        "initial_rank1_energy", "best_rank1_energy", "final_rank1_energy",
        "initial_center_ratio", "best_center_ratio", "final_center_ratio",
        "initial_effective_rank", "final_effective_rank", "initial_margin", "final_margin",
        "initial_normalized_margin", "final_normalized_margin", "initial_volume_cv", "final_volume_cv",
        "final_active_edge_clusters", "final_largest_edge_ratio",
        "final_active_node_clusters", "final_largest_node_ratio",
        "max_qtdq_condition_number", "runtime_seconds", "peak_gpu_memory_mb", "status",
        "initial_q_checksum", "initial_weight_checksum", "prototype_center_checksum",
    ]
    combined_rows = []
    comparison_rows = []
    run_data = {}

    for config, loss_type, orth_type in CONFIGS:
        for seed in SEEDS:
            run_dir = out_dir / config / f"seed{seed}"
            result = read_json(run_dir / "result.json")
            config_json = read_json(run_dir / "config.json")
            stages = stages_for(run_dir)
            metrics = read_csv(run_dir / "metrics.csv")
            init = initial_stage(stages)
            final = stages.get("final_epoch") or stages.get("epoch_20") or {}
            metric_by_epoch = {int(float(row["epoch"])): row for row in metrics if row.get("epoch")}

            init_row = {
                "loss_config": config,
                "seed": seed,
                "epoch": 0,
                "cluster_loss_type": loss_type,
                "orth_type": orth_type,
            }
            for key in combined_fields:
                if key not in init_row:
                    init_row[key] = stage_value(init, key)
            combined_rows.append(init_row)

            for epoch in range(1, 21):
                stage = stages.get(f"epoch_{epoch}", {})
                metric = metric_by_epoch.get(epoch, {})
                row = {
                    "loss_config": config,
                    "seed": seed,
                    "epoch": epoch,
                    "cluster_loss_type": loss_type,
                    "orth_type": orth_type,
                    "cut_loss": metric.get("cut_loss", stage_value(stage, "cut_loss")),
                    "orth_loss": metric.get("orth_loss", stage_value(stage, "orth_loss")),
                    "orth_original_loss": metric.get("orth_original_loss", stage_value(stage, "orth_original_loss")),
                    "orthqa_loss": metric.get("orthqa_loss", stage_value(stage, "orthqa_loss")),
                    "orth_original_value": metric.get("orth_original_value", stage_value(stage, "orth_original_value")),
                    "orthqa_value": metric.get("orthqa_value", stage_value(stage, "orthqa_value")),
                    "total_loss": metric.get("global_total_loss", ""),
                }
                for key in combined_fields:
                    if key in row:
                        continue
                    row[key] = metric.get(key, stage_value(stage, key))
                combined_rows.append(row)

            valid_metrics = [row for row in metrics if math.isfinite(finite(row.get("Macro_F1")))]
            best_metric = max(valid_metrics, key=lambda row: finite(row.get("Macro_F1")), default={})
            best_rank = min([finite(row.get("q_rank1_energy_ratio")) for row in valid_metrics] or [math.nan])
            best_center = max([finite(row.get("q_centered_to_total_energy_ratio")) for row in valid_metrics] or [math.nan])
            conditions = [number(row.get("qtdq_condition_number")) for row in valid_metrics]
            conditions = [value for value in conditions if not math.isnan(value)]
            model_init = config_json.get("model_init_info", {}) or {}
            row = {
                "config": config,
                "seed": seed,
                "initial_macro_f1": stage_value(init, "Macro_F1"),
                "best_macro_f1": result.get("best_metrics", {}).get("Macro_F1", best_metric.get("Macro_F1", "")),
                "best_epoch": result.get("best_epoch", best_metric.get("epoch", "")),
                "final_macro_f1": result.get("final_metrics", {}).get("Macro_F1", stage_value(final, "Macro_F1")),
                "best_nmi": best_metric.get("NMI", ""),
                "final_nmi": result.get("final_metrics", {}).get("NMI", stage_value(final, "NMI")),
                "best_ari": best_metric.get("ARI", ""),
                "final_ari": result.get("final_metrics", {}).get("ARI", stage_value(final, "ARI")),
                "initial_rank1_energy": stage_value(init, "q_rank1_energy_ratio"),
                "best_rank1_energy": best_rank,
                "final_rank1_energy": stage_value(final, "q_rank1_energy_ratio"),
                "initial_center_ratio": stage_value(init, "q_centered_to_total_energy_ratio"),
                "best_center_ratio": best_center,
                "final_center_ratio": stage_value(final, "q_centered_to_total_energy_ratio"),
                "initial_effective_rank": stage_value(init, "q_effective_rank"),
                "final_effective_rank": stage_value(final, "q_effective_rank"),
                "initial_margin": stage_value(init, "q_margin_mean"),
                "final_margin": stage_value(final, "q_margin_mean"),
                "initial_normalized_margin": stage_value(init, "q_normalized_margin_mean"),
                "final_normalized_margin": stage_value(final, "q_normalized_margin_mean"),
                "initial_volume_cv": stage_value(init, "cluster_volume_cv"),
                "final_volume_cv": stage_value(final, "cluster_volume_cv"),
                "final_active_edge_clusters": stage_value(final, "num_active_edge_clusters"),
                "final_largest_edge_ratio": stage_value(final, "largest_edge_cluster_ratio"),
                "final_active_node_clusters": stage_value(final, "num_active_node_clusters"),
                "final_largest_node_ratio": stage_value(final, "largest_node_cluster_ratio"),
                "max_qtdq_condition_number": max(conditions) if conditions else "",
                "runtime_seconds": result.get("runtime_seconds", ""),
                "peak_gpu_memory_mb": max([finite(row.get("peak_gpu_memory_mb"), 0.0) for row in metrics] or [0.0]),
                "status": result.get("status", "missing"),
                "initial_q_checksum": model_init.get("initial_Q_summary_checksum", ""),
                "initial_weight_checksum": model_init.get("initial_cluster_weight_checksum", ""),
                "prototype_center_checksum": model_init.get("prototype_center_checksum", ""),
            }
            comparison_rows.append(row)
            run_data[(config, seed)] = {
                "result": result,
                "config": config_json,
                "stages": stages,
                "metrics": metrics,
                "comparison": row,
            }

    write_csv(out_dir / "combined_summary.csv", combined_fields, combined_rows)
    write_csv(out_dir / "loss_comparison.csv", comparison_fields, comparison_rows)

    by_config = {config: [row for row in comparison_rows if row["config"] == config] for config, _, _ in CONFIGS}
    checksum_lines = []
    initialization_matched = True
    for seed in SEEDS:
        for key in ["initial_q_checksum", "initial_weight_checksum", "prototype_center_checksum"]:
            values = {row[key] for row in comparison_rows if int(row["seed"]) == seed}
            matched = len(values) == 1 and "" not in values
            initialization_matched = initialization_matched and matched
            checksum_lines.append(f"- seed {seed} {key}: {'matched' if matched else 'MISMATCH'}")

    c6_lines = []
    l1_reproduced = None
    if reference_dir and reference_dir.exists():
        checks = []
        for seed in SEEDS:
            ref_stages = stages_for(reference_dir / f"seed_{seed}")
            ref_final = ref_stages.get("final_epoch", {})
            new_epoch3 = run_data[("L1_trace_orth", seed)]["stages"].get("epoch_3", {})
            rank_delta = abs(finite(stage_value(new_epoch3, "q_rank1_energy_ratio")) - finite(stage_value(ref_final, "q_rank1_energy_ratio")))
            f1_delta = abs(finite(stage_value(new_epoch3, "Macro_F1")) - finite(stage_value(ref_final, "Macro_F1")))
            matched = rank_delta <= 1e-6 and f1_delta <= 1e-8
            checks.append(matched)
            c6_lines.append(
                f"- seed {seed}: epoch-3 rank1 delta={rank_delta:.3g}, Macro-F1 delta={f1_delta:.3g}, "
                f"{'matched' if matched else 'MISMATCH'}"
            )
        l1_reproduced = all(checks)

    def delta(left, right, key):
        return mean(by_config[right], key) - mean(by_config[left], key)

    effect_rows = [
        ("Orth to OrthQA under trace", "L1_trace_orth", "L2_trace_orthqa"),
        ("Orth to OrthQA under matrix", "L3_matrix_orth", "L4_matrix_orthqa"),
        ("Trace to matrix under orth", "L1_trace_orth", "L3_matrix_orth"),
        ("Trace to matrix under orthqa", "L2_trace_orthqa", "L4_matrix_orthqa"),
    ]
    lines = [
        "# ETGC Loss Formulation Validation",
        "",
        "Scope: School only, seeds 42/43, 20 epochs, fixed C6 representation/head/initialization.",
        "",
        "## Initialization Fairness",
        "",
        f"Overall initialization checksum audit: {'PASS' if initialization_matched else 'FAIL'}.",
        *checksum_lines,
        "",
        "## Per-Seed Final Summary",
        "",
        "| Config | Seed | Best F1 | Final F1 | Final NMI | Final ARI | Final Rank1 | Center Ratio | Eff Rank | Norm Margin | Volume CV | Edge Active | Node Active | Max QTDQ Cond |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['config']} | {row['seed']} | {fmt(row['best_macro_f1'])} | {fmt(row['final_macro_f1'])} | "
            f"{fmt(row['final_nmi'])} | {fmt(row['final_ari'])} | {fmt(row['final_rank1_energy'])} | "
            f"{fmt(row['final_center_ratio'])} | {fmt(row['final_effective_rank'])} | "
            f"{fmt(row['final_normalized_margin'])} | {fmt(row['final_volume_cv'])} | "
            f"{fmt(row['final_active_edge_clusters'])} | {fmt(row['final_active_node_clusters'])} | "
            f"{fmt(row['max_qtdq_condition_number'])} |"
        )

    lines.extend(["", "## C6 Reproduction", ""])
    if l1_reproduced is None:
        lines.append("C6 reference was not supplied; L1 reproduction is not established.")
    else:
        lines.append(f"L1 epoch 3 reproduces the previous C6 run: {'yes' if l1_reproduced else 'no'}.")
        lines.extend(c6_lines)

    lines.extend(["", "## Main Effects", ""])
    for label, left, right in effect_rows:
        lines.append(
            f"- {label}: delta final F1={fmt(delta(left, right, 'final_macro_f1'))}, "
            f"delta volume CV={fmt(delta(left, right, 'final_volume_cv'))}, "
            f"delta normalized margin={fmt(delta(left, right, 'final_normalized_margin'))}, "
            f"delta center ratio={fmt(delta(left, right, 'final_center_ratio'))}."
        )
    interaction_f1 = (
        delta("L3_matrix_orth", "L4_matrix_orthqa", "final_macro_f1")
        - delta("L1_trace_orth", "L2_trace_orthqa", "final_macro_f1")
    )
    lines.append(f"- F1 difference-in-differences interaction={fmt(interaction_f1)}.")

    best_f1 = max(CONFIGS, key=lambda item: mean(by_config[item[0]], "final_macro_f1"))[0]
    best_geometry = max(CONFIGS, key=lambda item: mean(by_config[item[0]], "final_center_ratio"))[0]
    best_margin = max(CONFIGS, key=lambda item: mean(by_config[item[0]], "final_normalized_margin"))[0]
    best_volume = min(CONFIGS, key=lambda item: mean(by_config[item[0]], "final_volume_cv"))[0]
    seed_ranges = {
        config: abs(finite(rows[0]["final_macro_f1"]) - finite(rows[1]["final_macro_f1"]))
        for config, rows in by_config.items() if len(rows) == 2
    }
    best_stability = min(seed_ranges, key=seed_ranges.get) if seed_ranges else "not available"
    low_margin = [
        f"{row['config']}/seed{row['seed']}"
        for row in comparison_rows
        if finite(row["final_normalized_margin"]) < 0.01
    ]

    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. L1 reproduces C6: {('yes' if l1_reproduced else 'no') if l1_reproduced is not None else 'not established'}.",
            "2. OrthQA versus orth: use both main-effect rows above; consistency requires the sign to agree under both cuts.",
            "3. Matrix Ncut versus trace-ratio: use both main-effect rows above; consistency requires the sign to agree under both penalties.",
            f"4. Best mean final Macro-F1: {best_f1}.",
            f"5. Best seed stability by final-F1 range: {best_stability}.",
            f"6. Best assignment geometry by centered ratio: {best_geometry}; best normalized margin: {best_margin}.",
            f"7. Best degree-weighted volume balance: {best_volume}.",
            "8. Low-margin risk runs: " + (", ".join(low_margin) if low_margin else "none at normalized-margin < 0.01"),
            "9. Matrix Ncut conditioning is reported per seed above; non-finite solves or extreme condition numbers must block selection.",
            "10. Can matrix_ncut + orthqa be fixed as the formal ETGC objective: not yet. This single-dataset two-seed ablation can select the next candidate, not establish a formal all-dataset objective.",
            "",
            "## Files",
            f"- combined_summary.csv: {out_dir / 'combined_summary.csv'}",
            f"- loss_comparison.csv: {out_dir / 'loss_comparison.csv'}",
            f"- diagnosis_report.md: {out_dir / 'diagnosis_report.md'}",
        ]
    )
    (out_dir / "diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failed = [row for row in comparison_rows if row["status"] != "success"]
    print(f"combined_summary={out_dir / 'combined_summary.csv'}")
    print(f"loss_comparison={out_dir / 'loss_comparison.csv'}")
    print(f"diagnosis_report={out_dir / 'diagnosis_report.md'}")
    raise SystemExit(1 if failed or not initialization_matched else 0)


if __name__ == "__main__":
    main()
