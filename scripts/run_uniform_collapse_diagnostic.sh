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
OUT_DIR="${ROOT_DIR}/logs/uniform_collapse_diagnostic/${RUN_TIMESTAMP}"
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

{
  echo "config_source_dir=${CONFIG_SOURCE_DIR}"
  for seed in 42 43; do
    path="${CONFIG_SOURCE_DIR}/D1_trace_only_seed${seed}/config.json"
    echo "seed_${seed}_config=${path}"
    if [[ -f "${path}" ]]; then
      "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys
path = sys.argv[1]
cfg = json.load(open(path, "r", encoding="utf-8"))
keys = [
    "dataset", "seed", "learning_rate", "lambda_prox", "lambda_edge_ncut", "lambda_proj",
    "lambda_orth", "lambda_bal", "node_emb_mode", "global_warmup_epochs", "forest_samples",
    "edge_neighbor_k", "edge_ppr_topk", "edge_ppr_method",
]
print(json.dumps({key: cfg.get(key) for key in keys}, sort_keys=True))
PY
    else
      echo "missing"
    fi
  done
} > "${OUT_DIR}/code_info/config_source.txt"

"${PYTHON_BIN}" - "${ROOT_DIR}" "${OUT_DIR}" "${CONFIG_SOURCE_DIR}" "${PYTHON_BIN}" "${DEVICE:-}" <<'PY'
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
config_source = Path(sys.argv[3])
python_bin = sys.argv[4]
device = sys.argv[5]

seed_list = [42, 43]


def read_source_config(seed):
    path = config_source / f"D1_trace_only_seed{seed}" / "config.json"
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        cfg = {}
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
        "ncut_scope": "global",
        "cluster_loss_type": "trace_mincut",
        "global_q_chunk_size": 8192,
        "global_ncut_row_block_size": 65536,
        "global_warmup_epochs": 0,
        "lambda_prox": 0.0,
        "lambda_edge_ncut": 0.5,
        "lambda_orth": 1.0,
        "lambda_proj": 0.0,
        "lambda_bal": 0.0,
        "node_emb_mode": "frozen",
        "node_emb_lr": 1e-5,
        "eval_every": 1,
        "save_embeddings": 0,
    }
    for key, value in fallback.items():
        cfg.setdefault(key, value)
    cfg["seed"] = seed
    cfg["dataset"] = "school"
    return cfg, path


def command_for(seed, run_dir):
    cfg, source_path = read_source_config(seed)
    args = {
        "dataset": cfg["dataset"],
        "directed": cfg["directed"],
        "device": device or cfg.get("device", "auto"),
        "seed": seed,
        "data_root": str(root / "dataset"),
        "emb_root": str(root / "emb"),
        "pretrain_emb_dir": str(root / "pretrain"),
        "feature_path": cfg.get("feature_path", ""),
        "cache_dir": str(root / "cache"),
        "batch_size": cfg["batch_size"],
        "epoch": 2,
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
        "ncut_scope": cfg["ncut_scope"],
        "cluster_loss_type": cfg["cluster_loss_type"],
        "global_q_chunk_size": cfg["global_q_chunk_size"],
        "global_ncut_row_block_size": cfg["global_ncut_row_block_size"],
        "global_warmup_epochs": cfg["global_warmup_epochs"],
        "quiet": 1,
        "lambda_prox": cfg["lambda_prox"],
        "lambda_edge_ncut": cfg["lambda_edge_ncut"],
        "lambda_orth": cfg["lambda_orth"],
        "lambda_proj": cfg["lambda_proj"],
        "lambda_bal": cfg["lambda_bal"],
        "node_emb_mode": cfg["node_emb_mode"],
        "node_emb_lr": cfg["node_emb_lr"],
        "diagnostic_stages": 0,
        "uniform_collapse_diagnostic": 1,
        "diagnostic_output_dir": str(run_dir),
        "diagnostic_only_first_epoch": 1,
        "output_dir": str(run_dir),
        "eval_every": cfg.get("eval_every", 1),
        "save_embeddings": 0,
    }
    cmd = [python_bin, "edge_main.py"]
    for key, value in args.items():
        if value == "":
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd, source_path


def run_one(seed):
    run_dir = out_dir / f"school_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd, source_path = command_for(seed, run_dir)
    (run_dir / "source_config_path.txt").write_text(str(source_path) + "\n", encoding="utf-8")
    if (run_dir / "diagnostic.json").exists():
        return 0
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("cmd=" + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


exit_codes = {}
for seed in seed_list:
    exit_codes[seed] = run_one(seed)


def load_summary_rows():
    rows = []
    for seed in seed_list:
        path = out_dir / f"school_seed{seed}" / "diagnostic_summary.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as reader:
            rows.extend(list(csv.DictReader(reader)))
    return rows


rows = load_summary_rows()
combined_path = out_dir / "combined_summary.csv"
if rows:
    fieldnames = list(rows[0].keys())
    with combined_path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(row)
else:
    combined_path.write_text("", encoding="utf-8")


def f(row, key, default=math.nan):
    try:
        value = row.get(key, "")
        if value in ("", "None", "null"):
            return default
        return float(value)
    except Exception:
        return default


by_seed_stage = {(int(row["seed"]), row["stage"]): row for row in rows if row.get("seed")}
lines = [
    "# Uniform Collapse Diagnostic",
    "",
    f"Output directory: {out_dir}",
    f"Config source: {config_source}",
    "",
    "## Summary",
    "",
    "| Seed | Stage | Edge Active | Edge Max | Node Active | Node Max | Margin Mean | Margin P99 | Logits Std | Uniform L2 | Entropy Gap | Cut Grad L2 | Orth Grad L2 | Cosine |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for seed in seed_list:
    for stage in ["initial_before_training", "before_first_global_update", "after_first_global_update"]:
        row = by_seed_stage.get((seed, stage))
        if not row:
            continue
        lines.append(
            f"| {seed} | {stage} | {row.get('num_active_edge_clusters','')} | {f(row,'largest_edge_cluster_ratio'):.6g} | "
            f"{row.get('num_active_node_clusters','')} | {f(row,'largest_node_cluster_ratio'):.6g} | "
            f"{f(row,'q_margin_mean'):.6g} | {f(row,'q_margin_p99'):.6g} | {f(row,'logits_global_std'):.6g} | "
            f"{f(row,'q_uniform_l2_mean'):.6g} | {f(row,'q_entropy_gap'):.6g} | "
            f"{f(row,'cut_cluster_head_grad_l2'):.6g} | {f(row,'orth_cluster_head_grad_l2'):.6g} | {f(row,'cut_orth_grad_cosine'):.6g} |"
        )

lines.extend(["", "## Answers", ""])
for seed in seed_list:
    initial = by_seed_stage.get((seed, "initial_before_training"), {})
    before = by_seed_stage.get((seed, "before_first_global_update"), {})
    after = by_seed_stage.get((seed, "after_first_global_update"), {})
    if not initial or not after:
        lines.append(f"- seed {seed}: missing diagnostic stages; exit_code={exit_codes.get(seed)}")
        continue
    initial_uniform = f(initial, "q_uniform_l2_mean")
    after_uniform = f(after, "q_uniform_l2_mean")
    delta_uniform = after_uniform - initial_uniform
    margin_p99 = f(initial, "q_margin_p99")
    logits_std = f(initial, "logits_global_std")
    cut_grad = f(before, "cut_cluster_head_grad_l2")
    orth_grad = f(before, "orth_cluster_head_grad_l2")
    cosine = f(before, "cut_orth_grad_cosine")
    edge_active = initial.get("num_active_edge_clusters", "")
    node_active = initial.get("num_active_node_clusters", "")
    near_uniform = initial_uniform < 1e-2 and f(initial, "q_entropy_gap") < 1e-3
    leaves_uniform = abs(delta_uniform) >= 1e-4
    lines.append(
        f"- seed {seed}: initial Q near uniform={near_uniform} "
        f"(uniform_l2={initial_uniform:.6g}, entropy_gap={f(initial,'q_entropy_gap'):.6g}); "
        f"after first global update leaves uniform clearly={leaves_uniform} "
        f"(after_l2={after_uniform:.6g}, delta={delta_uniform:.6g})."
    )
    lines.append(
        f"- seed {seed}: argmax(Q) active edge clusters={edge_active}, argmax(S) active node clusters={node_active}; "
        f"initial margin_p99={margin_p99:.6g}, logits_global_std={logits_std:.6g}."
    )
    lines.append(
        f"- seed {seed}: cut_grad_l2={cut_grad:.6g}, orth_grad_l2={orth_grad:.6g}, "
        f"cut_orth_grad_cosine={cosine:.6g}; near-zero thresholds are reported numerically rather than assumed."
    )

if all((seed, "initial_before_training") in by_seed_stage for seed in seed_list):
    i42 = by_seed_stage[(42, "initial_before_training")]
    i43 = by_seed_stage[(43, "initial_before_training")]
    lines.append(
        "- seed comparison: "
        f"seed42 margin_mean={f(i42,'q_margin_mean'):.6g}, edge_active={i42.get('num_active_edge_clusters','')}, node_active={i42.get('num_active_node_clusters','')}; "
        f"seed43 margin_mean={f(i43,'q_margin_mean'):.6g}, edge_active={i43.get('num_active_edge_clusters','')}, node_active={i43.get('num_active_node_clusters','')}. "
        "If hard active counts differ while uniform/margin/logit scales remain tiny, the difference is driven by argmax-level perturbations."
    )

lines.extend(
    [
        "",
        "## Files",
        f"- combined_summary.csv: {combined_path}",
        f"- diagnosis_report.md: {out_dir / 'diagnosis_report.md'}",
    ]
)
(out_dir / "diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"OUT_DIR={out_dir}")
print(f"COMBINED_SUMMARY={combined_path}")
print(f"DIAGNOSIS_REPORT={out_dir / 'diagnosis_report.md'}")
for seed, code in exit_codes.items():
    print(f"seed={seed} exit_code={code}")
sys.exit(0 if all(code == 0 for code in exit_codes.values()) else 1)
PY
