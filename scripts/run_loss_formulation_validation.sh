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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/loss_formulation_validation/${RUN_TIMESTAMP}}"
DEVICE_ARG="${DEVICE:-cuda:0}"
DATASET_ARG="${DATASET:-school}"
FOREST_SAMPLES_ARG="${FOREST_SAMPLES:-5}"
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
import torch
print("torch_version=" + str(torch.__version__))
print("cuda_available=" + str(torch.cuda.is_available()))
print("cuda_device=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
PY
  env | sort | grep -E '^(DATASET|DEVICE|FOREST_SAMPLES|CUDA_VISIBLE_DEVICES|OPENBLAS_NUM_THREADS|OMP_NUM_THREADS|TEMPORAL_FOREST_)=' || true
} > "${OUT_DIR}/code_info/environment.txt"

"${PYTHON_BIN}" - "${OUT_DIR}/code_info/common_config.json" "${ROOT_DIR}" "${ASSET_ROOT:-}" "${DATASET_ARG}" "${FOREST_SAMPLES_ARG}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = Path(sys.argv[2])
explicit = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
dataset = sys.argv[4]
forest_samples = int(sys.argv[5])
server_root = Path("/mnt/data/lin-lab/lmx/projects/my_project/ETGC")
asset_root = next(
    (candidate for candidate in [explicit, root, server_root] if candidate and (candidate / "dataset/school/school.txt").exists()),
    root,
)
common = {
    "method": "ETGC",
    "purpose": "loss formulation validation with fixed C6 configuration",
    "dataset": dataset,
    "seeds": [42, 43],
    "epoch": 20,
    "fixed_configuration": {
        "cluster_head_type": "legacy_mlp",
        "edge_encoder_mode": "mlp",
        "time_feature_mode": "history",
        "cluster_output_bias_mode": "zero",
        "cluster_input_norm": "layernorm",
        "cluster_init_mode": "prototype",
        "node_emb_mode": "frozen",
        "learning_rate": 1e-4,
        "batch_size": 512,
        "prototype_sample_size": 20000,
        "prototype_lloyd_iters": 10,
        "edge_ppr_method": "temporal_state_forest",
        "forest_samples": forest_samples,
        "forest_seed": 20260725,
        "edge_neighbor_k": -1,
        "edge_ppr_topk": -1,
        "affinity_sparsify": "symmetric_union_knn",
        "alpha": 0.2,
        "T": 4,
        "beta": 5.0,
        "lambda_prox": 0.0,
        "lambda_proj": 0.0,
        "lambda_bal": 0.0,
        "lambda_edge_ncut": 0.5,
        "lambda_orth": 1.0,
        "global_warmup_epochs": 0,
        "prox_warmup_epochs": 0,
    },
    "varied_factors": ["cluster_loss_type", "orth_type"],
    "loss_matrix": [
        {"config": "L1_trace_orth", "cluster_loss_type": "legacy_trace_ratio", "orth_type": "orth"},
        {"config": "L2_trace_orthqa", "cluster_loss_type": "legacy_trace_ratio", "orth_type": "orthqa"},
        {"config": "L3_matrix_orth", "cluster_loss_type": "matrix_ncut", "orth_type": "orth"},
        {"config": "L4_matrix_orthqa", "cluster_loss_type": "matrix_ncut", "orth_type": "orthqa"},
    ],
    "asset_root": str(asset_root),
    "save_embeddings": 0,
    "checkpoint": "disabled/not produced by edge_main.py",
}
path.write_text(json.dumps(common, indent=2, sort_keys=True), encoding="utf-8")
PY

printf 'loss_formulation_out_dir=%s\n' "${OUT_DIR}"

"${PYTHON_BIN}" - "${ROOT_DIR}" "${OUT_DIR}" "${PYTHON_BIN}" "${DEVICE_ARG}" "${RESUME}" "${ASSET_ROOT:-}" "${DATASET_ARG}" "${FOREST_SAMPLES_ARG}" <<'PY'
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
python_bin = sys.argv[3]
device = sys.argv[4]
resume = bool(int(sys.argv[5]))
explicit_asset_root = Path(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
dataset = sys.argv[7]
forest_samples = int(sys.argv[8])
server_asset_root = Path("/mnt/data/lin-lab/lmx/projects/my_project/ETGC")
asset_root = next(
    (candidate for candidate in [explicit_asset_root, root, server_asset_root] if candidate and (candidate / "dataset/school/school.txt").exists()),
    root,
)

configs = [
    ("L1_trace_orth", "legacy_trace_ratio", "orth"),
    ("L2_trace_orthqa", "legacy_trace_ratio", "orthqa"),
    ("L3_matrix_orth", "matrix_ncut", "orth"),
    ("L4_matrix_orthqa", "matrix_ncut", "orthqa"),
]
seeds = [42, 43]
allowed = {"config.json", "metrics.csv", "diagnostic.json", "result.json", "train.log"}


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def complete(run_dir):
    result = read_json(run_dir / "result.json")
    if result.get("status") != "success":
        return False
    metrics = run_dir / "metrics.csv"
    diagnostic = read_json(run_dir / "diagnostic.json")
    stages = diagnostic.get("stages", {}) if isinstance(diagnostic.get("stages"), dict) else {}
    if not metrics.exists() or not (run_dir / "config.json").exists():
        return False
    epoch_rows = max(0, len(metrics.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    return epoch_rows >= 20 and bool(stages.get("epoch_20")) and bool(stages.get("final_epoch"))


def cleanup(run_dir):
    if not run_dir.exists():
        return
    for item in run_dir.iterdir():
        if item.name in allowed:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def command(loss_type, orth_type, seed, run_dir):
    args = {
        "dataset": dataset,
        "directed": 0,
        "device": device,
        "seed": seed,
        "model_seed": seed,
        "prototype_seed": seed,
        "forest_seed": 20260725,
        "data_root": str(asset_root / "dataset"),
        "emb_root": str(asset_root / "emb"),
        "pretrain_emb_dir": str(asset_root / "pretrain"),
        "cache_dir": str(asset_root / "cache"),
        "batch_size": 512,
        "epoch": 20,
        "learning_rate": 1e-4,
        "edge_dim": 128,
        "time_dim": 32,
        "edge_hidden_dim": 128,
        "cluster_hidden_dim": 64,
        "time_feature_mode": "history",
        "edge_encoder_mode": "mlp",
        "cluster_head_type": "legacy_mlp",
        "alpha": 0.2,
        "T": 4,
        "beta": 5.0,
        "edge_neighbor_k": -1,
        "edge_ppr_topk": -1,
        "affinity_sparsify": "symmetric_union_knn",
        "edge_ppr_method": "temporal_state_forest",
        "forest_samples": forest_samples,
        "ncut_scope": "global",
        "cluster_loss_type": loss_type,
        "orth_type": orth_type,
        "global_q_chunk_size": 8192,
        "global_ncut_row_block_size": 65536,
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
        "node_emb_lr": 1e-5,
        "prox_similarity_mode": "event_dot",
        "cluster_output_bias_mode": "zero",
        "cluster_input_norm": "layernorm",
        "cluster_init_mode": "prototype",
        "prototype_sample_size": 20000,
        "prototype_lloyd_iters": 10,
        "overnight_diagnostic": 0,
        "loss_formulation_diagnostic": 1,
        "diagnostic_epochs": ",".join(str(epoch) for epoch in range(1, 21)),
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
        cmd.extend([f"--{key}", str(value)])
    return cmd


failed = []
for config, loss_type, orth_type in configs:
    for seed in seeds:
        run_dir = out_dir / config / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if resume and complete(run_dir):
            print(f"skip config={config} seed={seed}", flush=True)
            cleanup(run_dir)
            continue
        cmd = command(loss_type, orth_type, seed, run_dir)
        print(f"run config={config} seed={seed}", flush=True)
        started = time.time()
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            log.write("cmd=" + " ".join(cmd) + "\n")
            proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"\nscript_runtime_seconds={time.time() - started:.6f}\n")
            log.write(f"script_exit_code={proc.returncode}\n")
        if proc.returncode != 0:
            tail = "\n".join((run_dir / "train.log").read_text(encoding="utf-8", errors="ignore").splitlines()[-30:])
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "config": config,
                        "seed": seed,
                        "exit_code": proc.returncode,
                        "error_summary": tail[-2000:],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            failed.append((config, seed))
        cleanup(run_dir)

print(f"run_failures={failed}")
PY

RUN_STATUS=$?
C6_ARGS=()
if [[ -n "${C6_REFERENCE_DIR:-}" ]]; then
  C6_ARGS=(--c6-reference-dir "${C6_REFERENCE_DIR}")
fi
"${PYTHON_BIN}" scripts/summarize_loss_formulation_validation.py "${OUT_DIR}" "${C6_ARGS[@]}"
SUMMARY_STATUS=$?

printf 'OUT_DIR=%s\n' "${OUT_DIR}"
if [[ ${RUN_STATUS} -ne 0 || ${SUMMARY_STATUS} -ne 0 ]]; then
  exit 1
fi
