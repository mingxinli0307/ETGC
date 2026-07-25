#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No python or python3 executable found." >&2
  exit 127
fi

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TEMPORAL_FOREST_WORKERS="${TEMPORAL_FOREST_WORKERS:-1}"
export TEMPORAL_FOREST_CHUNK_SAMPLES="${TEMPORAL_FOREST_CHUNK_SAMPLES:-1}"
export TEMPORAL_FOREST_COMBINE_CHUNKS="${TEMPORAL_FOREST_COMBINE_CHUNKS:-8}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${ROOT_DIR}/logs/rank1_cause_validation/${RUN_TIMESTAMP}"
CONFIG_SOURCE_DIR="${CONFIG_SOURCE_DIR:-${ROOT_DIR}/logs/trace_mincut_global/20260724_002532/phase1_diagnosis}"
mkdir -p "${OUT_DIR}/code_info"

{
  git rev-parse HEAD 2>/dev/null || true
  git log -1 --oneline 2>/dev/null || true
} > "${OUT_DIR}/code_info/commit.txt"

{
  echo "which_python=$(command -v "${PYTHON_BIN}" || true)"
  "${PYTHON_BIN}" --version 2>&1
  "${PYTHON_BIN}" - <<'PY'
try:
    import torch
    print("torch_version=" + str(torch.__version__))
    print("cuda_available=" + str(torch.cuda.is_available()))
    print("cuda_device=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
except Exception as exc:
    print("torch_probe_error=" + repr(exc))
PY
  env | sort | grep -E '^(DEVICE|CUDA_VISIBLE_DEVICES|OPENBLAS_NUM_THREADS|OMP_NUM_THREADS|TEMPORAL_FOREST_)=' || true
} > "${OUT_DIR}/code_info/environment.txt"

cat > "${OUT_DIR}/code_info/common_config.json" <<EOF
{
  "dataset": "school",
  "seeds": [42, 43],
  "epoch": 3,
  "cluster_loss_type": "trace_mincut",
  "orth_type": "orth",
  "ncut_scope": "global",
  "lambda_prox": 0.0,
  "lambda_edge_ncut": 0.5,
  "lambda_orth": 1.0,
  "lambda_proj": 0.0,
  "lambda_bal": 0.0,
  "node_emb_mode": "frozen",
  "global_warmup_epochs": 0,
  "edge_ppr_method": "temporal_state_forest",
  "edge_ppr_topk": -1,
  "uniform_collapse_diagnostic": 1,
  "diagnostic_only_first_epoch": 0,
  "quiet": 1,
  "orthqa_dataset_experiments": false
}
EOF

"${PYTHON_BIN}" - "${ROOT_DIR}" "${OUT_DIR}" "${CONFIG_SOURCE_DIR}" "${PYTHON_BIN}" "${DEVICE:-}" <<'PY'
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
config_source = Path(sys.argv[3])
python_bin = sys.argv[4]
device = sys.argv[5]

CONFIGS = [
    ("C0_baseline", "default", "none", "random"),
    ("C1_zero_bias", "zero", "none", "random"),
    ("C2_no_bias", "none", "none", "random"),
    ("C3_layernorm", "default", "layernorm", "random"),
    ("C4_no_bias_layernorm", "none", "layernorm", "random"),
    ("C5_prototype", "zero", "none", "prototype"),
    ("C6_layernorm_prototype", "zero", "layernorm", "prototype"),
    ("C7_no_bias_layernorm_prototype", "none", "layernorm", "prototype"),
]
SEEDS = [42, 43]


def load_source_config(seed):
    path = config_source / f"D1_trace_only_seed{seed}" / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    fallback = {
        "dataset": "school",
        "directed": 0,
        "seed": seed,
        "batch_size": 512,
        "learning_rate": 1e-4,
        "edge_dim": 128,
        "time_dim": 32,
        "edge_hidden_dim": 128,
        "cluster_hidden_dim": 64,
        "alpha": 0.2,
        "T": 4,
        "beta": 5.0,
        "edge_neighbor_k": -1,
        "edge_ppr_topk": -1,
        "edge_ppr_method": "temporal_state_forest",
        "forest_samples": 5,
        "global_q_chunk_size": 8192,
        "global_ncut_row_block_size": 65536,
        "node_emb_lr": 1e-5,
    }
    for key, value in fallback.items():
        cfg.setdefault(key, value)
    cfg["dataset"] = "school"
    cfg["seed"] = seed
    return cfg


def run_dir_for(config_name, seed):
    return out_dir / config_name / f"seed_{seed}"


def command_for(config_name, seed, bias_mode, input_norm, init_mode, run_dir):
    cfg = load_source_config(seed)
    args = {
        "dataset": "school",
        "directed": cfg["directed"],
        "device": device or cfg.get("device", "auto"),
        "seed": seed,
        "data_root": str(root / "dataset"),
        "emb_root": str(root / "emb"),
        "pretrain_emb_dir": str(root / "pretrain"),
        "feature_path": cfg.get("feature_path", ""),
        "cache_dir": str(root / "cache"),
        "batch_size": cfg["batch_size"],
        "epoch": 3,
        "learning_rate": cfg["learning_rate"],
        "edge_dim": cfg["edge_dim"],
        "time_dim": cfg["time_dim"],
        "edge_hidden_dim": cfg["edge_hidden_dim"],
        "cluster_hidden_dim": cfg["cluster_hidden_dim"],
        "alpha": cfg["alpha"],
        "T": cfg["T"],
        "beta": cfg["beta"],
        "edge_neighbor_k": cfg["edge_neighbor_k"],
        "edge_ppr_topk": cfg["edge_ppr_topk"],
        "edge_ppr_method": cfg["edge_ppr_method"],
        "forest_samples": cfg["forest_samples"],
        "ncut_scope": "global",
        "cluster_loss_type": "trace_mincut",
        "orth_type": "orth",
        "global_q_chunk_size": cfg["global_q_chunk_size"],
        "global_ncut_row_block_size": cfg["global_ncut_row_block_size"],
        "global_warmup_epochs": 0,
        "quiet": 1,
        "lambda_prox": 0.0,
        "lambda_edge_ncut": 0.5,
        "lambda_orth": 1.0,
        "lambda_proj": 0.0,
        "lambda_bal": 0.0,
        "node_emb_mode": "frozen",
        "node_emb_lr": cfg["node_emb_lr"],
        "cluster_output_bias_mode": bias_mode,
        "cluster_input_norm": input_norm,
        "cluster_init_mode": init_mode,
        "prototype_sample_size": 20000,
        "prototype_lloyd_iters": 10,
        "diagnostic_stages": 0,
        "uniform_collapse_diagnostic": 1,
        "diagnostic_output_dir": str(run_dir),
        "diagnostic_only_first_epoch": 0,
        "output_dir": str(run_dir),
        "eval_every": 1,
        "save_embeddings": 0,
    }
    cmd = [python_bin, "edge_main.py"]
    for key, value in args.items():
        if value == "":
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def is_complete(run_dir):
    result = run_dir / "result.json"
    diagnostic = run_dir / "diagnostic.json"
    if not result.exists() or not diagnostic.exists():
        return False
    try:
        return json.loads(result.read_text(encoding="utf-8")).get("status") == "success"
    except Exception:
        return False


def write_failure(run_dir, config_name, seed, error_summary, exit_code):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "config": config_name,
        "seed": seed,
        "error_summary": error_summary,
        "exit_code": exit_code,
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_one(config_name, seed, bias_mode, input_norm, init_mode):
    run_dir = run_dir_for(config_name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_complete(run_dir):
        return
    cmd = command_for(config_name, seed, bias_mode, input_norm, init_mode, run_dir)
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("cmd=" + " ".join(cmd) + "\n")
        started = time.time()
        proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nscript_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"script_exit_code={proc.returncode}\n")
    if proc.returncode != 0:
        tail = "\n".join((run_dir / "train.log").read_text(encoding="utf-8", errors="ignore").splitlines()[-12:])
        write_failure(run_dir, config_name, seed, tail[:800], proc.returncode)


for config_name, bias_mode, input_norm, init_mode in CONFIGS:
    for seed in SEEDS:
        print(f"run config={config_name} seed={seed}", flush=True)
        run_one(config_name, seed, bias_mode, input_norm, init_mode)


def finite(value, default=math.nan):
    try:
        if value in ("", None):
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as reader:
        return list(csv.DictReader(reader))


def run_status_and_runtime(run_dir):
    result_path = run_dir / "result.json"
    status = "missing"
    runtime = math.nan
    best = {}
    final = {}
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status = result.get("status", "failed")
            runtime = finite(result.get("runtime_seconds"))
            best = result.get("best_metrics", {}) or {}
            final = result.get("final_metrics", {}) or {}
        except Exception:
            status = "failed"
    metric_rows = read_csv_rows(run_dir / "metrics.csv")
    peak = max([finite(row.get("peak_gpu_memory_mb"), 0.0) for row in metric_rows] or [0.0])
    return status, runtime, peak, best, final


combined_fields = [
    "config",
    "seed",
    "stage",
    "orth_type",
    "cluster_output_bias_mode",
    "cluster_input_norm",
    "cluster_init_mode",
    "num_active_edge_clusters",
    "largest_edge_cluster_ratio",
    "num_active_node_clusters",
    "largest_node_cluster_ratio",
    "q_rank1_energy_ratio",
    "q_second_energy_ratio",
    "q_effective_rank",
    "q_numerical_rank",
    "q_centered_energy",
    "q_centered_effective_rank",
    "q_centered_numerical_rank",
    "q_margin_mean",
    "q_margin_p99",
    "q_uniform_l2_mean",
    "q_entropy_gap",
    "feature_common_to_variation_ratio_before_norm",
    "feature_common_to_variation_ratio_after_norm",
    "logits_cluster_mean_std",
    "logits_within_cluster_event_std",
    "logits_bias_to_event_variation_ratio",
    "output_bias_l2",
    "cluster_output_weight_l2",
    "cut_loss",
    "orth_loss",
    "cut_cluster_head_grad_l2",
    "orth_cluster_head_grad_l2",
    "cut_orth_grad_cosine",
    "ACC",
    "NMI",
    "ARI",
    "Macro_F1",
    "runtime_seconds",
    "peak_gpu_memory_mb",
    "status",
]

combined_rows = []
for config_name, bias_mode, input_norm, init_mode in CONFIGS:
    for seed in SEEDS:
        run_dir = run_dir_for(config_name, seed)
        status, runtime, peak, best, final = run_status_and_runtime(run_dir)
        for row in read_csv_rows(run_dir / "diagnostic_summary.csv"):
            out = {key: "" for key in combined_fields}
            out.update({key: row.get(key, "") for key in combined_fields})
            out["config"] = config_name
            out["seed"] = seed
            out["orth_type"] = "orth"
            out["cluster_output_bias_mode"] = bias_mode
            out["cluster_input_norm"] = input_norm
            out["cluster_init_mode"] = init_mode
            out["runtime_seconds"] = runtime
            out["peak_gpu_memory_mb"] = peak
            out["status"] = status
            combined_rows.append(out)

combined_path = out_dir / "combined_summary.csv"
with combined_path.open("w", encoding="utf-8", newline="") as writer:
    w = csv.DictWriter(writer, fieldnames=combined_fields)
    w.writeheader()
    for row in combined_rows:
        w.writerow(row)


def rows_for(config, stage):
    return [row for row in combined_rows if row["config"] == config and row["stage"] == stage and row["status"] == "success"]


def mean_for(config, stage, key):
    vals = [finite(row.get(key)) for row in rows_for(config, stage)]
    vals = [x for x in vals if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else math.nan


factor_fields = [
    "config",
    "bias_mode",
    "input_norm",
    "init_mode",
    "mean_after_rank1",
    "delta_after_rank1_vs_C0",
    "mean_final_rank1",
    "delta_final_rank1_vs_C0",
    "mean_final_centered_energy",
    "mean_final_effective_rank",
    "mean_final_active_edge_clusters",
    "mean_final_largest_edge_ratio",
    "mean_final_active_node_clusters",
    "mean_final_largest_node_ratio",
    "mean_final_bias_event_ratio",
    "mean_final_macro_f1",
    "status",
]
c0_after_rank1 = mean_for("C0_baseline", "after_first_global_update", "q_rank1_energy_ratio")
c0_final_rank1 = mean_for("C0_baseline", "final_epoch", "q_rank1_energy_ratio")
factor_rows = []
for config_name, bias_mode, input_norm, init_mode in CONFIGS:
    after_rank1 = mean_for(config_name, "after_first_global_update", "q_rank1_energy_ratio")
    final_rank1 = mean_for(config_name, "final_epoch", "q_rank1_energy_ratio")
    statuses = {row["status"] for row in combined_rows if row["config"] == config_name}
    factor_rows.append(
        {
            "config": config_name,
            "bias_mode": bias_mode,
            "input_norm": input_norm,
            "init_mode": init_mode,
            "mean_after_rank1": after_rank1,
            "delta_after_rank1_vs_C0": after_rank1 - c0_after_rank1 if math.isfinite(after_rank1) and math.isfinite(c0_after_rank1) else math.nan,
            "mean_final_rank1": final_rank1,
            "delta_final_rank1_vs_C0": final_rank1 - c0_final_rank1 if math.isfinite(final_rank1) and math.isfinite(c0_final_rank1) else math.nan,
            "mean_final_centered_energy": mean_for(config_name, "final_epoch", "q_centered_energy"),
            "mean_final_effective_rank": mean_for(config_name, "final_epoch", "q_effective_rank"),
            "mean_final_active_edge_clusters": mean_for(config_name, "final_epoch", "num_active_edge_clusters"),
            "mean_final_largest_edge_ratio": mean_for(config_name, "final_epoch", "largest_edge_cluster_ratio"),
            "mean_final_active_node_clusters": mean_for(config_name, "final_epoch", "num_active_node_clusters"),
            "mean_final_largest_node_ratio": mean_for(config_name, "final_epoch", "largest_node_cluster_ratio"),
            "mean_final_bias_event_ratio": mean_for(config_name, "final_epoch", "logits_bias_to_event_variation_ratio"),
            "mean_final_macro_f1": mean_for(config_name, "final_epoch", "Macro_F1"),
            "status": "success" if statuses == {"success"} else ",".join(sorted(statuses)),
        }
    )

factor_path = out_dir / "factor_comparison.csv"
with factor_path.open("w", encoding="utf-8", newline="") as writer:
    w = csv.DictWriter(writer, fieldnames=factor_fields)
    w.writeheader()
    for row in factor_rows:
        w.writerow(row)


def fmt(x):
    return "nan" if not math.isfinite(finite(x)) else f"{float(x):.6g}"


def best_by(key, reverse=False):
    valid = [row for row in factor_rows if math.isfinite(finite(row.get(key)))]
    if not valid:
        return None
    return sorted(valid, key=lambda r: finite(r[key]), reverse=reverse)[0]


best_rank = best_by("mean_final_rank1", reverse=False)
best_center = best_by("mean_final_centered_energy", reverse=True)
best_macro = best_by("mean_final_macro_f1", reverse=True)

single_factors = ["C1_zero_bias", "C2_no_bias", "C3_layernorm", "C5_prototype"]
single_rank = sorted(
    [row for row in factor_rows if row["config"] in single_factors],
    key=lambda r: finite(r.get("mean_final_rank1")),
)
best_single = single_rank[0] if single_rank else None

lines = [
    "# Rank-1 Cause Validation",
    "",
    f"Output directory: {out_dir}",
    "All C0-C7 runs use orth_type=orth. orthqa is implemented and tested only; no orthqa dataset experiment is run here.",
    "",
    "## Factor Summary",
    "",
    "| Config | Bias | Norm | Init | Final Rank1 | Final Centered Energy | Final Eff Rank | Edge Active | Edge Max | Node Active | Bias/Event | Macro-F1 | Status |",
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in factor_rows:
    lines.append(
        f"| {row['config']} | {row['bias_mode']} | {row['input_norm']} | {row['init_mode']} | "
        f"{fmt(row['mean_final_rank1'])} | {fmt(row['mean_final_centered_energy'])} | "
        f"{fmt(row['mean_final_effective_rank'])} | {fmt(row['mean_final_active_edge_clusters'])} | "
        f"{fmt(row['mean_final_largest_edge_ratio'])} | {fmt(row['mean_final_active_node_clusters'])} | "
        f"{fmt(row['mean_final_bias_event_ratio'])} | {fmt(row['mean_final_macro_f1'])} | {row['status']} |"
    )

lines.extend(["", "## Questions", ""])
lines.append(
    f"1. Random output bias: compare C0 vs C1/C2. C0 final rank1={fmt(c0_final_rank1)}, "
    f"C1={fmt(mean_for('C1_zero_bias','final_epoch','q_rank1_energy_ratio'))}, "
    f"C2={fmt(mean_for('C2_no_bias','final_epoch','q_rank1_energy_ratio'))}."
)
lines.append(
    f"2. zero bias vs no bias: C1 final rank1={fmt(mean_for('C1_zero_bias','final_epoch','q_rank1_energy_ratio'))}, "
    f"C2 final rank1={fmt(mean_for('C2_no_bias','final_epoch','q_rank1_energy_ratio'))}; "
    f"C1 edge max={fmt(mean_for('C1_zero_bias','final_epoch','largest_edge_cluster_ratio'))}, "
    f"C2 edge max={fmt(mean_for('C2_no_bias','final_epoch','largest_edge_cluster_ratio'))}."
)
lines.append(
    f"3. LayerNorm input common component: C0 after-norm common/variation={fmt(mean_for('C0_baseline','after_model_initialization','feature_common_to_variation_ratio_after_norm'))}, "
    f"C3={fmt(mean_for('C3_layernorm','after_model_initialization','feature_common_to_variation_ratio_after_norm'))}."
)
lines.append(
    f"4. LayerNorm logits variation: C0 final bias/event={fmt(mean_for('C0_baseline','final_epoch','logits_bias_to_event_variation_ratio'))}, "
    f"C3={fmt(mean_for('C3_layernorm','final_epoch','logits_bias_to_event_variation_ratio'))}."
)
lines.append(
    f"5. Prototype rank1 effect: C5 after-prototype rank1={fmt(mean_for('C5_prototype','after_prototype_initialization','q_rank1_energy_ratio'))}, "
    f"C5 final rank1={fmt(mean_for('C5_prototype','final_epoch','q_rank1_energy_ratio'))}."
)
lines.append(
    f"6. Prototype after first update: C5 after-first rank1={fmt(mean_for('C5_prototype','after_first_global_update','q_rank1_energy_ratio'))}, "
    f"C6 after-first rank1={fmt(mean_for('C6_layernorm_prototype','after_first_global_update','q_rank1_energy_ratio'))}, "
    f"C7 after-first rank1={fmt(mean_for('C7_no_bias_layernorm_prototype','after_first_global_update','q_rank1_energy_ratio'))}."
)
if best_single:
    lines.append(f"7. Best single factor by final rank1: {best_single['config']} with final rank1={fmt(best_single['mean_final_rank1'])}.")
if best_rank:
    lines.append(f"8. Best combination by final rank1: {best_rank['config']} with final rank1={fmt(best_rank['mean_final_rank1'])}.")
lines.append(
    "9. Hard clusters without true rank break should be identified by combined_summary rows where active edge clusters increase while rank1 remains near 1 and centered energy remains small."
)
lines.append(
    "10. Seed consistency is assessed by per-seed rows in combined_summary; a factor is robust only if both seeds move rank1, centered energy, edge max ratio, and bias/event ratio in the same direction."
)
if best_center:
    lines.append(f"Centered-energy leader: {best_center['config']} with centered energy={fmt(best_center['mean_final_centered_energy'])}.")
if best_macro:
    lines.append(f"Macro-F1 leader: {best_macro['config']} with Macro-F1={fmt(best_macro['mean_final_macro_f1'])}.")

lines.extend(
    [
        "",
        "## Orthqa Implementation",
        "",
        "Function: edge_orthqa_penalty_global(Q_all, degree, eps=1e-12).",
        "Formula: (sqrt(K) - sum_k sqrt(sum_i degree_i Q_ik^2 + eps) / sqrt(sum_i degree_i + eps)) / (sqrt(K) - 1).",
        "Complexity: O(MK); no dense D_E, no incidence matrix, and no Q/degree detach.",
        "Tests cover dense reference, balanced one-hot near zero, rank-one Q near one, unbalanced one-hot positive, soft backward, K=1 exception, nonpositive volume exception, and default orth branch compatibility.",
        "No orthqa dataset experiment was run in C0-C7; all runs pass --orth_type orth.",
        "",
        "## Files",
        f"- combined_summary.csv: {combined_path}",
        f"- factor_comparison.csv: {factor_path}",
        f"- diagnosis_report.md: {out_dir / 'diagnosis_report.md'}",
    ]
)
(out_dir / "diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"OUT_DIR={out_dir}")
print(f"COMBINED_SUMMARY={combined_path}")
print(f"FACTOR_COMPARISON={factor_path}")
print(f"DIAGNOSIS_REPORT={out_dir / 'diagnosis_report.md'}")
failed = [row for row in factor_rows if row["status"] != "success"]
sys.exit(1 if failed else 0)
PY
