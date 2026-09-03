#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


CELLS = [(42, 42), (42, 43), (43, 42), (43, 43)]


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


def stage_map(run_dir):
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
    return stage.get(aliases.get(key), default) if aliases.get(key) else default


def fmt(value):
    value = number(value)
    return "nan" if math.isnan(value) else f"{value:.6g}"


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    fields = [
        "model_seed", "prototype_seed", "source", "initial_macro_f1", "best_macro_f1", "best_epoch",
        "final_macro_f1", "final_nmi", "final_ari", "initial_rank1_energy", "final_rank1_energy",
        "initial_center_ratio", "final_center_ratio", "initial_effective_rank", "final_effective_rank",
        "initial_normalized_margin", "final_normalized_margin", "initial_volume_cv", "final_volume_cv",
        "final_active_edge_clusters", "final_active_node_clusters", "max_qtdq_condition_number",
        "runtime_seconds", "status", "prototype_feature_checksum", "prototype_center_checksum",
        "initial_q_checksum",
    ]
    rows = []
    by_cell = {}
    for model_seed, prototype_seed in CELLS:
        run_dir = out_dir / f"model{model_seed}_prototype{prototype_seed}"
        config = read_json(run_dir / "config.json")
        result = read_json(run_dir / "result.json")
        metrics = read_csv(run_dir / "metrics.csv")
        stages = stage_map(run_dir)
        init = stages.get("after_cluster_initialization") or stages.get("after_prototype_initialization") or {}
        final = stages.get("final_epoch") or stages.get("epoch_20") or {}
        valid = [row for row in metrics if math.isfinite(number(row.get("Macro_F1")))]
        best = max(valid, key=lambda row: number(row.get("Macro_F1")), default={})
        conditions = [number(row.get("qtdq_condition_number")) for row in valid]
        conditions = [value for value in conditions if math.isfinite(value)]
        init_info = config.get("model_init_info", {}) or {}
        row = {
            "model_seed": model_seed,
            "prototype_seed": prototype_seed,
            "source": "reused" if model_seed == prototype_seed else "new",
            "initial_macro_f1": stage_value(init, "Macro_F1"),
            "best_macro_f1": result.get("best_metrics", {}).get("Macro_F1", best.get("Macro_F1", "")),
            "best_epoch": result.get("best_epoch", best.get("epoch", "")),
            "final_macro_f1": result.get("final_metrics", {}).get("Macro_F1", stage_value(final, "Macro_F1")),
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
            "max_qtdq_condition_number": max(conditions) if conditions else "",
            "runtime_seconds": result.get("runtime_seconds", ""),
            "status": result.get("status", "missing"),
            "prototype_feature_checksum": init_info.get("prototype_feature_checksum", ""),
            "prototype_center_checksum": init_info.get("prototype_center_checksum", ""),
            "initial_q_checksum": init_info.get("initial_Q_summary_checksum", ""),
        }
        rows.append(row)
        by_cell[(model_seed, prototype_seed)] = row

    write_csv(out_dir / "seed_source_comparison.csv", fields, rows)
    y = {(m, p): number(by_cell[(m, p)]["final_macro_f1"]) for m, p in CELLS}
    prototype_effect_m42 = y[(42, 43)] - y[(42, 42)]
    prototype_effect_m43 = y[(43, 43)] - y[(43, 42)]
    model_effect_p42 = y[(43, 42)] - y[(42, 42)]
    model_effect_p43 = y[(43, 43)] - y[(42, 43)]
    prototype_main = 0.5 * (prototype_effect_m42 + prototype_effect_m43)
    model_main = 0.5 * (model_effect_p42 + model_effect_p43)
    interaction = y[(43, 43)] - y[(43, 42)] - y[(42, 43)] + y[(42, 42)]

    feature_checks = {}
    for model_seed in (42, 43):
        feature_checks[model_seed] = (
            by_cell[(model_seed, 42)]["prototype_feature_checksum"]
            == by_cell[(model_seed, 43)]["prototype_feature_checksum"]
            != ""
        )
    centers_differ = {
        model_seed: by_cell[(model_seed, 42)]["prototype_center_checksum"]
        != by_cell[(model_seed, 43)]["prototype_center_checksum"]
        for model_seed in (42, 43)
    }
    dominant = "model/training seed" if abs(model_main) > abs(prototype_main) else "prototype seed"
    report = [
        "# ETGC L3 Seed-Source Validation",
        "",
        "Scope: School, matrix_ncut + orth, fixed C6 configuration, 20 epochs.",
        "This 2x2 is a causal diagnostic with one run per cell, not a variance estimate.",
        "",
        "## Results",
        "",
        "| Model seed | Prototype seed | Source | Initial F1 | Best F1 | Best epoch | Final F1 | Final NMI | Final ARI | Final Rank1 | Center ratio | Norm margin | Volume CV |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['model_seed']} | {row['prototype_seed']} | {row['source']} | "
            f"{fmt(row['initial_macro_f1'])} | {fmt(row['best_macro_f1'])} | {row['best_epoch']} | "
            f"{fmt(row['final_macro_f1'])} | {fmt(row['final_nmi'])} | {fmt(row['final_ari'])} | "
            f"{fmt(row['final_rank1_energy'])} | {fmt(row['final_center_ratio'])} | "
            f"{fmt(row['final_normalized_margin'])} | {fmt(row['final_volume_cv'])} |"
        )
    report += [
        "",
        "## Factor Contrasts",
        "",
        f"- Prototype 42 to 43 at model seed 42: delta final F1={fmt(prototype_effect_m42)}.",
        f"- Prototype 42 to 43 at model seed 43: delta final F1={fmt(prototype_effect_m43)}.",
        f"- Model/training 42 to 43 at prototype seed 42: delta final F1={fmt(model_effect_p42)}.",
        f"- Model/training 42 to 43 at prototype seed 43: delta final F1={fmt(model_effect_p43)}.",
        f"- Two-level prototype main effect={fmt(prototype_main)}.",
        f"- Two-level model/training main effect={fmt(model_main)}.",
        f"- Difference-in-differences interaction={fmt(interaction)}.",
        f"- Larger absolute two-level main effect: {dominant}.",
        "",
        "## Fairness Audit",
        "",
        f"- Model seed 42 prototype feature checksum fixed across prototype seeds: {feature_checks[42]}.",
        f"- Model seed 43 prototype feature checksum fixed across prototype seeds: {feature_checks[43]}.",
        f"- Model seed 42 prototype center checksum changes with prototype seed: {centers_differ[42]}.",
        f"- Model seed 43 prototype center checksum changes with prototype seed: {centers_differ[43]}.",
        "",
        "## Interpretation Boundary",
        "",
        "The two model seeds also select different encoder hidden representations and training trajectories. "
        "The prototype effect is conditional on each representation. More seeds are required before a stability claim.",
    ]
    (out_dir / "diagnosis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"seed_source_comparison={out_dir / 'seed_source_comparison.csv'}")
    print(f"diagnosis_report={out_dir / 'diagnosis_report.md'}")


if __name__ == "__main__":
    main()
