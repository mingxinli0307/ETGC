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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/rank1_cause_validation/${RUN_TIMESTAMP}}"
CONFIG_SOURCE_DIR="${CONFIG_SOURCE_DIR:-${ROOT_DIR}/logs/trace_mincut_global/20260724_002532/phase1_diagnosis}"
DEVICE_ARG="${DEVICE:-cuda:0}"
RESUME=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-resume)
      RESUME=0
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUT_DIR}/code_info"

{
  git rev-parse HEAD 2>/dev/null || true
  git log -1 --oneline 2>/dev/null || true
} > "${OUT_DIR}/code_info/commit.txt"

{
  echo "which_python=$(command -v "${PYTHON_BIN}" || true)"
  "${PYTHON_BIN}" --version 2>&1 || true
  "${PYTHON_BIN}" - <<'PY' || true
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

"${PYTHON_BIN}" - "${OUT_DIR}/code_info/common_config.json" <<'PY'
import json
import sys

path = sys.argv[1]
common = {
    "method": "ETGC",
    "purpose": "rank-1 collapse cause validation",
    "dataset": "school",
    "seeds": [42, 43],
    "epoch": 3,
    "edge_encoder_mode": "mlp",
    "time_feature_mode": "history",
    "cluster_head_type": "legacy_mlp",
    "ncut_scope": "global",
    "cluster_loss_type": "legacy_trace_ratio",
    "cut_formulation": "legacy scalar trace-ratio cut (historical trace_mincut alias)",
    "orth_type": "orth",
    "lambda_prox": 0.0,
    "lambda_edge_ncut": 0.5,
    "lambda_orth": 1.0,
    "lambda_proj": 0.0,
    "lambda_bal": 0.0,
    "node_emb_mode": "frozen",
    "global_warmup_epochs": 0,
    "prox_warmup_epochs": 0,
    "uniform_collapse_diagnostic": 1,
    "diagnostic_only_first_epoch": 0,
    "save_embeddings": 0,
    "checkpoint": "disabled/not produced by edge_main.py",
    "factors_varied": [
        "cluster_output_bias_mode",
        "cluster_input_norm",
        "cluster_init_mode",
    ],
}
with open(path, "w", encoding="utf-8") as writer:
    json.dump(common, writer, indent=2, sort_keys=True)
PY

printf 'launcher_out_dir=%s\n' "${OUT_DIR}" | tee "${OUT_DIR}/launcher.log"
printf 'config_source_dir=%s\n' "${CONFIG_SOURCE_DIR}" | tee -a "${OUT_DIR}/launcher.log"

"${PYTHON_BIN}" - "${ROOT_DIR}" "${OUT_DIR}" "${CONFIG_SOURCE_DIR}" "${PYTHON_BIN}" "${DEVICE_ARG}" "${RESUME}" <<'PY' 2>&1 | tee -a "${OUT_DIR}/launcher.log"
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
config_source = Path(sys.argv[3])
python_bin = sys.argv[4]
device = sys.argv[5]
resume = bool(int(sys.argv[6]))

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
FOREST_SEED = 20260725
ALLOWED_RUN_FILES = {"config.json", "diagnostic.json", "metrics.csv", "result.json", "train.log"}
REQUIRED_STAGES = [
    "after_model_initialization",
    "before_first_global_update",
    "after_first_global_update",
    "final_epoch",
]
COMBINED_STAGES = [
    "after_model_initialization",
    "after_prototype_initialization",
    "after_cluster_initialization",
    "before_first_global_update",
    "after_first_global_update",
    "final_epoch",
]


def load_source_config(seed):
    path = config_source / f"D1_trace_only_seed{seed}" / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    defaults = {
        "directed": 0,
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
        "feature_path": "",
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    cfg["dataset"] = "school"
    cfg["seed"] = int(seed)
    return cfg


def run_dir_for(config_name, seed):
    return out_dir / config_name / f"seed_{seed}"


def command_for(config_name, seed, bias_mode, input_norm, init_mode, run_dir):
    cfg = load_source_config(seed)
    args = {
        "dataset": "school",
        "directed": cfg["directed"],
        "device": device,
        "seed": seed,
        "model_seed": seed,
        "prototype_seed": seed,
        "forest_seed": FOREST_SEED,
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
        "time_feature_mode": "history",
        "edge_encoder_mode": "mlp",
        "cluster_head_type": "legacy_mlp",
        "alpha": cfg["alpha"],
        "T": cfg["T"],
        "beta": cfg["beta"],
        "edge_neighbor_k": cfg["edge_neighbor_k"],
        "edge_ppr_topk": cfg["edge_ppr_topk"],
        "edge_ppr_method": cfg["edge_ppr_method"],
        "forest_samples": cfg["forest_samples"],
        "ncut_scope": "global",
        "cluster_loss_type": "legacy_trace_ratio",
        "orth_type": "orth",
        "global_q_chunk_size": cfg["global_q_chunk_size"],
        "global_ncut_row_block_size": cfg["global_ncut_row_block_size"],
        "global_warmup_epochs": 0,
        "prox_warmup_epochs": 0,
        "quiet": 1,
        "lambda_prox": 0.0,
        "lambda_edge_ncut": 0.5,
        "lambda_orth": 1.0,
        "lambda_proj": 0.0,
        "lambda_bal": 0.0,
        "lambda_node_anchor": 0.0,
        "node_emb_mode": "frozen",
        "node_emb_lr": cfg["node_emb_lr"],
        "prox_similarity_mode": "event_dot",
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


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def diagnostic_stages(run_dir):
    payload = read_json(Path(run_dir) / "diagnostic.json")
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    for name in COMBINED_STAGES:
        if name not in stages and isinstance(payload.get(name), dict):
            stages[name] = payload[name]
    return stages


def is_complete(run_dir, init_mode):
    run_dir = Path(run_dir)
    result = read_json(run_dir / "result.json")
    if result.get("status") != "success":
        return False
    if not (run_dir / "config.json").exists() or not (run_dir / "metrics.csv").exists():
        return False
    stages = diagnostic_stages(run_dir)
    required = list(REQUIRED_STAGES)
    if init_mode == "prototype":
        required.append("after_prototype_initialization")
    return all(stages.get(stage) for stage in required)


def cleanup_run_dir(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    for item in run_dir.iterdir():
        if item.name in ALLOWED_RUN_FILES:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def write_failure(run_dir, config_name, seed, error_summary, exit_code):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "config": config_name,
        "seed": int(seed),
        "error_summary": str(error_summary)[:1200],
        "exit_code": int(exit_code),
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_one(config_name, seed, bias_mode, input_norm, init_mode):
    run_dir = run_dir_for(config_name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and is_complete(run_dir, init_mode):
        print(f"skip config={config_name} seed={seed}", flush=True)
        cleanup_run_dir(run_dir)
        return
    cmd = command_for(config_name, seed, bias_mode, input_norm, init_mode, run_dir)
    print(f"run config={config_name} seed={seed}", flush=True)
    started = time.time()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("cmd=" + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nscript_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"script_exit_code={proc.returncode}\n")
    if proc.returncode != 0:
        tail = "\n".join((run_dir / "train.log").read_text(encoding="utf-8", errors="ignore").splitlines()[-24:])
        write_failure(run_dir, config_name, seed, tail, proc.returncode)
    cleanup_run_dir(run_dir)


def finite(value, default=math.nan):
    try:
        if value in ("", None, "None", "null"):
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as reader:
        return list(csv.DictReader(reader))


def metric_peak(run_dir):
    rows = read_csv_rows(Path(run_dir) / "metrics.csv")
    return max([finite(row.get("peak_gpu_memory_mb"), 0.0) for row in rows] or [0.0])


def status_runtime_best_final(run_dir):
    result = read_json(Path(run_dir) / "result.json")
    return (
        result.get("status", "missing"),
        finite(result.get("runtime_seconds")),
        result.get("best_metrics", {}) or {},
        result.get("final_metrics", {}) or {},
    )


def stage_metric(stage, key):
    if not stage:
        return ""
    if key == "feature_common_to_variation_ratio":
        return stage.get(
            "feature_common_to_variation_ratio_after_norm",
            stage.get("feature_common_to_variation_ratio_before_norm", ""),
        )
    return stage.get(key, "")


def initial_stage(stages):
    return stages.get("after_cluster_initialization") or stages.get("after_prototype_initialization") or stages.get("after_model_initialization") or {}


for config_name, bias_mode, input_norm, init_mode in CONFIGS:
    for seed in SEEDS:
        run_one(config_name, seed, bias_mode, input_norm, init_mode)

core_fields = [
    "q_rank1_energy_ratio",
    "q_second_energy_ratio",
    "q_effective_rank",
    "q_numerical_rank",
    "q_centered_energy",
    "q_centered_effective_rank",
    "q_centered_numerical_rank",
    "num_active_edge_clusters",
    "largest_edge_cluster_ratio",
    "num_active_node_clusters",
    "largest_node_cluster_ratio",
    "q_margin_mean",
    "q_margin_p99",
    "q_uniform_l2_mean",
    "q_entropy_gap",
    "feature_common_to_variation_ratio",
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
]

combined_fields = [
    "config",
    "seed",
    "stage",
    "cluster_output_bias_mode",
    "cluster_input_norm",
    "cluster_init_mode",
    "status",
    "runtime_seconds",
    "peak_gpu_memory_mb",
] + core_fields
combined_rows = []
run_index = {}
for config_name, bias_mode, input_norm, init_mode in CONFIGS:
    for seed in SEEDS:
        run_dir = run_dir_for(config_name, seed)
        status, runtime, best, final = status_runtime_best_final(run_dir)
        peak = metric_peak(run_dir)
        stages = diagnostic_stages(run_dir)
        run_index[(config_name, seed)] = {
            "status": status,
            "runtime": runtime,
            "peak": peak,
            "best": best,
            "final": final,
            "stages": stages,
            "bias_mode": bias_mode,
            "input_norm": input_norm,
            "init_mode": init_mode,
            "run_dir": run_dir,
        }
        for stage_name in COMBINED_STAGES:
            stage = stages.get(stage_name)
            if not stage:
                continue
            row = {
                "config": config_name,
                "seed": int(seed),
                "stage": stage_name,
                "cluster_output_bias_mode": bias_mode,
                "cluster_input_norm": input_norm,
                "cluster_init_mode": init_mode,
                "status": status,
                "runtime_seconds": runtime,
                "peak_gpu_memory_mb": peak,
            }
            for key in core_fields:
                row[key] = stage_metric(stage, key)
            combined_rows.append(row)

combined_path = out_dir / "combined_summary.csv"
with combined_path.open("w", encoding="utf-8", newline="") as writer:
    writer_obj = csv.DictWriter(writer, fieldnames=combined_fields)
    writer_obj.writeheader()
    for row in combined_rows:
        writer_obj.writerow({key: row.get(key, "") for key in combined_fields})

factor_fields = [
    "config",
    "seed",
    "cluster_output_bias_mode",
    "cluster_input_norm",
    "cluster_init_mode",
    "initial_rank1_energy",
    "after_first_rank1_energy",
    "final_rank1_energy",
    "initial_centered_energy",
    "after_first_centered_energy",
    "final_centered_energy",
    "initial_effective_rank",
    "after_first_effective_rank",
    "final_effective_rank",
    "initial_centered_effective_rank",
    "after_first_centered_effective_rank",
    "final_centered_effective_rank",
    "initial_active_edge_clusters",
    "final_active_edge_clusters",
    "initial_largest_edge_ratio",
    "final_largest_edge_ratio",
    "initial_active_node_clusters",
    "final_active_node_clusters",
    "initial_largest_node_ratio",
    "final_largest_node_ratio",
    "initial_bias_event_ratio",
    "final_bias_event_ratio",
    "best_macro_f1",
    "final_macro_f1",
    "runtime_seconds",
    "peak_gpu_memory_mb",
    "status",
]


def pick(stage, key):
    return stage_metric(stage or {}, key)


factor_rows = []
for (config_name, seed), info in run_index.items():
    stages = info["stages"]
    init = initial_stage(stages)
    after = stages.get("after_first_global_update", {})
    final = stages.get("final_epoch", {})
    row = {
        "config": config_name,
        "seed": int(seed),
        "cluster_output_bias_mode": info["bias_mode"],
        "cluster_input_norm": info["input_norm"],
        "cluster_init_mode": info["init_mode"],
        "initial_rank1_energy": pick(init, "q_rank1_energy_ratio"),
        "after_first_rank1_energy": pick(after, "q_rank1_energy_ratio"),
        "final_rank1_energy": pick(final, "q_rank1_energy_ratio"),
        "initial_centered_energy": pick(init, "q_centered_energy"),
        "after_first_centered_energy": pick(after, "q_centered_energy"),
        "final_centered_energy": pick(final, "q_centered_energy"),
        "initial_effective_rank": pick(init, "q_effective_rank"),
        "after_first_effective_rank": pick(after, "q_effective_rank"),
        "final_effective_rank": pick(final, "q_effective_rank"),
        "initial_centered_effective_rank": pick(init, "q_centered_effective_rank"),
        "after_first_centered_effective_rank": pick(after, "q_centered_effective_rank"),
        "final_centered_effective_rank": pick(final, "q_centered_effective_rank"),
        "initial_active_edge_clusters": pick(init, "num_active_edge_clusters"),
        "final_active_edge_clusters": pick(final, "num_active_edge_clusters"),
        "initial_largest_edge_ratio": pick(init, "largest_edge_cluster_ratio"),
        "final_largest_edge_ratio": pick(final, "largest_edge_cluster_ratio"),
        "initial_active_node_clusters": pick(init, "num_active_node_clusters"),
        "final_active_node_clusters": pick(final, "num_active_node_clusters"),
        "initial_largest_node_ratio": pick(init, "largest_node_cluster_ratio"),
        "final_largest_node_ratio": pick(final, "largest_node_cluster_ratio"),
        "initial_bias_event_ratio": pick(init, "logits_bias_to_event_variation_ratio"),
        "final_bias_event_ratio": pick(final, "logits_bias_to_event_variation_ratio"),
        "best_macro_f1": info["best"].get("Macro_F1", ""),
        "final_macro_f1": info["final"].get("Macro_F1", ""),
        "runtime_seconds": info["runtime"],
        "peak_gpu_memory_mb": info["peak"],
        "status": info["status"],
    }
    factor_rows.append(row)

factor_path = out_dir / "factor_comparison.csv"
with factor_path.open("w", encoding="utf-8", newline="") as writer:
    writer_obj = csv.DictWriter(writer, fieldnames=factor_fields)
    writer_obj.writeheader()
    for row in factor_rows:
        writer_obj.writerow({key: row.get(key, "") for key in factor_fields})


def rows_for(config):
    return [row for row in factor_rows if row["config"] == config and row["status"] == "success"]


def mean_for(config, key):
    vals = [finite(row.get(key)) for row in rows_for(config)]
    vals = [value for value in vals if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else math.nan


def both_seed_improved(config, reference="C0_baseline"):
    rows = rows_for(config)
    if len(rows) != 2 or len(rows_for(reference)) != 2:
        return False
    ref_by_seed = {int(row["seed"]): row for row in rows_for(reference)}
    for row in rows:
        ref = ref_by_seed.get(int(row["seed"]))
        if not ref:
            return False
        if not (finite(row["final_rank1_energy"]) < finite(ref["final_rank1_energy"]) - 1e-4):
            return False
        if not (finite(row["final_centered_energy"]) > finite(ref["final_centered_energy"]) + 1e-8):
            return False
        if not (finite(row["final_effective_rank"]) > finite(ref["final_effective_rank"]) + 1e-4):
            return False
        if not (finite(row["after_first_rank1_energy"]) < 0.999):
            return False
        if not (finite(row["final_rank1_energy"]) < 0.999):
            return False
    return True


def fmt(value):
    x = finite(value)
    return "nan" if not math.isfinite(x) else f"{x:.6g}"


def avg_delta(config, key, reference="C0_baseline"):
    return mean_for(config, key) - mean_for(reference, key)


def collapse_like(row):
    return (
        finite(row.get("final_rank1_energy")) >= 0.999
        and finite(row.get("final_centered_energy"), 1.0) <= 1e-6
    )


c0_rows = rows_for("C0_baseline")
c0_reproduced = len(c0_rows) == 2 and all(collapse_like(row) for row in c0_rows)
single_factor_configs = ["C1_zero_bias", "C2_no_bias", "C3_layernorm", "C5_prototype"]
rank_candidates = [row for row in factor_rows if row["status"] == "success"]
best_single = min(
    [row for row in rank_candidates if row["config"] in single_factor_configs],
    key=lambda row: finite(row.get("final_rank1_energy"), math.inf),
    default=None,
)
best_combo = min(
    [row for row in rank_candidates if row["config"] not in ["C0_baseline"]],
    key=lambda row: finite(row.get("final_rank1_energy"), math.inf),
    default=None,
)
false_hard = [
    row
    for row in factor_rows
    if finite(row.get("final_active_edge_clusters"), 0) > 1
    and finite(row.get("final_rank1_energy"), 0) >= 0.999
    and finite(row.get("final_centered_energy"), 1.0) <= 1e-6
]
global_pullback = [
    row
    for row in factor_rows
    if finite(row.get("initial_rank1_energy")) < 0.999
    and finite(row.get("after_first_rank1_energy")) >= 0.999
]
eligible_next = [cfg for cfg, *_ in CONFIGS if cfg != "C0_baseline" and both_seed_improved(cfg)]

lines = [
    "# Rank-1 Collapse Cause Validation",
    "",
    f"Output directory: {out_dir}",
    "Common configuration: School, seeds 42/43, 3 epochs, legacy_mlp, mlp history-time edge encoder, legacy scalar trace-ratio cut, orth penalty, lambda_prox=0, lambda_proj=0, lambda_bal=0, frozen node embeddings.",
    "",
    "## Per-Seed Factor Summary",
    "",
    "| Config | Seed | Init R1 | After First R1 | Final R1 | Init Centered | Final Centered | Init Eff Rank | Final Eff Rank | Edge Active | Edge Max | Node Active | Node Max | Bias/Event | Best F1 | Final F1 | Status |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in factor_rows:
    lines.append(
        f"| {row['config']} | {row['seed']} | {fmt(row['initial_rank1_energy'])} | "
        f"{fmt(row['after_first_rank1_energy'])} | {fmt(row['final_rank1_energy'])} | "
        f"{fmt(row['initial_centered_energy'])} | {fmt(row['final_centered_energy'])} | "
        f"{fmt(row['initial_effective_rank'])} | {fmt(row['final_effective_rank'])} | "
        f"{fmt(row['final_active_edge_clusters'])} | {fmt(row['final_largest_edge_ratio'])} | "
        f"{fmt(row['final_active_node_clusters'])} | {fmt(row['final_largest_node_ratio'])} | "
        f"{fmt(row['final_bias_event_ratio'])} | {fmt(row['best_macro_f1'])} | "
        f"{fmt(row['final_macro_f1'])} | {row['status']} |"
    )

lines.extend(["", "## Required Answers", ""])
lines.append(
    "1. C0 collapse reproduction: "
    + (
        "yes, both seeds satisfy the strict high-rank1/low-centered-energy check."
        if c0_reproduced
        else "no under the strict check; downstream causal claims should be treated as not established unless the numeric C0 rows are judged to match the earlier failure."
    )
)
lines.append(
    f"2. Random output bias: C0 final R1 mean={fmt(mean_for('C0_baseline','final_rank1_energy'))}, "
    f"C1 zero-bias={fmt(mean_for('C1_zero_bias','final_rank1_energy'))}, "
    f"C2 no-bias={fmt(mean_for('C2_no_bias','final_rank1_energy'))}; centered-energy deltas vs C0 are "
    f"C1={fmt(avg_delta('C1_zero_bias','final_centered_energy'))}, C2={fmt(avg_delta('C2_no_bias','final_centered_energy'))}."
)
lines.append(
    f"3. Zero bias vs no bias: C1 final R1={fmt(mean_for('C1_zero_bias','final_rank1_energy'))}, "
    f"C2 final R1={fmt(mean_for('C2_no_bias','final_rank1_energy'))}; "
    f"C1 final bias/event={fmt(mean_for('C1_zero_bias','final_bias_event_ratio'))}, "
    f"C2 final bias/event={fmt(mean_for('C2_no_bias','final_bias_event_ratio'))}."
)
lines.append(
    f"4. LayerNorm common component: C0 initial centered={fmt(mean_for('C0_baseline','initial_centered_energy'))}, "
    f"C3={fmt(mean_for('C3_layernorm','initial_centered_energy'))}; "
    f"C2={fmt(mean_for('C2_no_bias','initial_centered_energy'))}, "
    f"C4={fmt(mean_for('C4_no_bias_layernorm','initial_centered_energy'))}."
)
lines.append(
    f"5. LayerNorm logits variation: C0 final bias/event={fmt(mean_for('C0_baseline','final_bias_event_ratio'))}, "
    f"C3={fmt(mean_for('C3_layernorm','final_bias_event_ratio'))}; "
    f"C2={fmt(mean_for('C2_no_bias','final_bias_event_ratio'))}, "
    f"C4={fmt(mean_for('C4_no_bias_layernorm','final_bias_event_ratio'))}."
)
lines.append(
    f"6. Prototype effective rank: C1 final eff-rank={fmt(mean_for('C1_zero_bias','final_effective_rank'))}, "
    f"C5={fmt(mean_for('C5_prototype','final_effective_rank'))}; "
    f"C3={fmt(mean_for('C3_layernorm','final_effective_rank'))}, "
    f"C6={fmt(mean_for('C6_layernorm_prototype','final_effective_rank'))}; "
    f"C4={fmt(mean_for('C4_no_bias_layernorm','final_effective_rank'))}, "
    f"C7={fmt(mean_for('C7_no_bias_layernorm_prototype','final_effective_rank'))}."
)
lines.append(
    "7. Best single factor by lowest final rank1: "
    + (f"{best_single['config']} seed {best_single['seed']} final R1={fmt(best_single['final_rank1_energy'])}." if best_single else "not available.")
)
lines.append(
    "8. Best combination by lowest final rank1: "
    + (f"{best_combo['config']} seed {best_combo['seed']} final R1={fmt(best_combo['final_rank1_energy'])}." if best_combo else "not available.")
)
lines.append(
    "9. Hard-cluster-only false improvements: "
    + (", ".join(f"{row['config']}/seed{row['seed']}" for row in false_hard) if false_hard else "none under the strict rank1>=0.999 and centered_energy<=1e-6 check.")
)
lines.append(
    "10. First global update pullback: "
    + (", ".join(f"{row['config']}/seed{row['seed']}" for row in global_pullback) if global_pullback else "no config moved from non-rank1 initialization back to strict rank1 after the first global update.")
)
seed_trends = []
for cfg, *_ in CONFIGS:
    if cfg == "C0_baseline":
        continue
    seed_trends.append(f"{cfg}:{'consistent' if both_seed_improved(cfg) else 'not_consistent_or_not_strong'}")
lines.append("11. Seed consistency: " + "; ".join(seed_trends) + ".")
if c0_reproduced:
    likely = []
    if mean_for("C1_zero_bias", "final_rank1_energy") < mean_for("C0_baseline", "final_rank1_energy") - 1e-4 or mean_for("C2_no_bias", "final_rank1_energy") < mean_for("C0_baseline", "final_rank1_energy") - 1e-4:
        likely.append("A output bias")
    if mean_for("C3_layernorm", "final_rank1_energy") < mean_for("C0_baseline", "final_rank1_energy") - 1e-4 or mean_for("C4_no_bias_layernorm", "final_rank1_energy") < mean_for("C2_no_bias", "final_rank1_energy") - 1e-4:
        likely.append("B representation common component")
    if mean_for("C5_prototype", "final_rank1_energy") < mean_for("C1_zero_bias", "final_rank1_energy") - 1e-4:
        likely.append("C random initialization symmetry")
    if global_pullback:
        likely.append("D global loss pullback")
    lines.append("12. Cause classification: " + (", ".join(likely) if likely else "E/mixed or inconclusive from thresholds") + ".")
else:
    lines.append("12. Cause classification: not asserted because C0 did not pass strict reproduction.")

lines.extend(["", "## Next-Stage Gate", ""])
if eligible_next:
    lines.append("At least one C1-C7 config satisfies the strict two-seed non-rank1 gate: " + ", ".join(eligible_next) + ".")
    lines.append("Loss formulation validation can be considered next, subject to user approval.")
else:
    lines.append("No C1-C7 config satisfies the strict two-seed non-rank1 gate. Do not run matrix_ncut/orthqa 2x2 yet.")

lines.extend(
    [
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

exit "${PIPESTATUS[0]}"
