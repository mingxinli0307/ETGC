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
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/l3_seed_source_validation/${RUN_TIMESTAMP}}"
REFERENCE_DIR="${LOSS_VALIDATION_DIR:-}"
DEVICE_ARG="${DEVICE:-cuda:0}"
RESUME=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-resume) RESUME=0; shift ;;
    --resume) RESUME=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${REFERENCE_DIR}" ]]; then
  echo "LOSS_VALIDATION_DIR must point to the completed phase-2 result directory." >&2
  exit 2
fi

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
  env | sort | grep -E '^(DEVICE|CUDA_VISIBLE_DEVICES|OPENBLAS_NUM_THREADS|OMP_NUM_THREADS|TEMPORAL_FOREST_)=' || true
} > "${OUT_DIR}/code_info/environment.txt"

"${PYTHON_BIN}" - "${OUT_DIR}/code_info/common_config.json" "${ROOT_DIR}" "${ASSET_ROOT:-}" "${REFERENCE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = Path(sys.argv[2])
explicit = Path(sys.argv[3]) if sys.argv[3] else None
reference = Path(sys.argv[4]).resolve()
server_root = Path("/mnt/data/lin-lab/lmx/projects/my_project/ETGC")
asset_root = next(
    (candidate for candidate in [explicit, root, server_root] if candidate and (candidate / "dataset/school/school.txt").exists()),
    root,
)
common = {
    "method": "ETGC",
    "purpose": "L3 model-training seed by prototype seed source validation",
    "dataset": "school",
    "epoch": 20,
    "objective": {"cluster_loss_type": "matrix_ncut", "orth_type": "orth"},
    "seed_matrix": [
        {"model_seed": 42, "prototype_seed": 42, "source": "reused"},
        {"model_seed": 42, "prototype_seed": 43, "source": "new"},
        {"model_seed": 43, "prototype_seed": 42, "source": "new"},
        {"model_seed": 43, "prototype_seed": 43, "source": "reused"},
    ],
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
        "forest_samples": 5,
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
    "varied_factors": ["model_seed", "prototype_seed"],
    "training_seed_semantics": "edge_main sets args.seed=model_seed before trainer construction",
    "phase2_reference_dir": str(reference),
    "asset_root": str(asset_root),
    "save_embeddings": 0,
    "checkpoint": "disabled/not produced by edge_main.py",
}
path.write_text(json.dumps(common, indent=2, sort_keys=True), encoding="utf-8")
PY

printf 'l3_seed_source_out_dir=%s\n' "${OUT_DIR}"

"${PYTHON_BIN}" - "${ROOT_DIR}" "${OUT_DIR}" "${PYTHON_BIN}" "${DEVICE_ARG}" "${RESUME}" "${ASSET_ROOT:-}" "${REFERENCE_DIR}" <<'PY'
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
explicit_asset_root = Path(sys.argv[6]) if sys.argv[6] else None
reference_dir = Path(sys.argv[7]).resolve()
server_asset_root = Path("/mnt/data/lin-lab/lmx/projects/my_project/ETGC")
asset_root = next(
    (candidate for candidate in [explicit_asset_root, root, server_asset_root] if candidate and (candidate / "dataset/school/school.txt").exists()),
    root,
)
allowed = {"config.json", "metrics.csv", "diagnostic.json", "result.json", "train.log"}
cells = [(42, 42, "reused"), (42, 43, "new"), (43, 42, "new"), (43, 43, "reused")]


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def complete(run_dir):
    result = read_json(run_dir / "result.json")
    diagnostic = read_json(run_dir / "diagnostic.json")
    stages = diagnostic.get("stages", {}) if isinstance(diagnostic.get("stages"), dict) else {}
    metrics = run_dir / "metrics.csv"
    if result.get("status") != "success" or not metrics.exists() or not (run_dir / "config.json").exists():
        return False
    rows = max(0, len(metrics.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    return rows >= 20 and bool(stages.get("epoch_20")) and bool(stages.get("final_epoch"))


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


def copy_reused(model_seed, run_dir):
    source = reference_dir / "L3_matrix_orth" / f"seed{model_seed}"
    if not complete(source):
        raise RuntimeError(f"Incomplete phase-2 reference: {source}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in allowed:
        shutil.copy2(source / name, run_dir / name)
    cleanup(run_dir)


def command(model_seed, prototype_seed, run_dir):
    args = {
        "dataset": "school", "directed": 0, "device": device,
        "seed": model_seed, "model_seed": model_seed,
        "prototype_seed": prototype_seed, "forest_seed": 20260725,
        "data_root": str(asset_root / "dataset"), "emb_root": str(asset_root / "emb"),
        "pretrain_emb_dir": str(asset_root / "pretrain"), "cache_dir": str(asset_root / "cache"),
        "batch_size": 512, "epoch": 20, "learning_rate": 1e-4,
        "edge_dim": 128, "time_dim": 32, "edge_hidden_dim": 128, "cluster_hidden_dim": 64,
        "time_feature_mode": "history", "edge_encoder_mode": "mlp", "cluster_head_type": "legacy_mlp",
        "alpha": 0.2, "T": 4, "beta": 5.0, "edge_neighbor_k": -1, "edge_ppr_topk": -1,
        "affinity_sparsify": "symmetric_union_knn", "edge_ppr_method": "temporal_state_forest",
        "forest_samples": 5, "ncut_scope": "global", "cluster_loss_type": "matrix_ncut",
        "orth_type": "orth", "global_q_chunk_size": 8192, "global_ncut_row_block_size": 65536,
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
        "output_dir": str(run_dir), "eval_every": 1, "save_embeddings": 0,
    }
    cmd = [python_bin, "edge_main.py"]
    for key, value in args.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


failed = []
for model_seed, prototype_seed, source in cells:
    cell = f"model{model_seed}_prototype{prototype_seed}"
    run_dir = out_dir / cell
    if source == "reused":
        print(f"reuse cell={cell}", flush=True)
        try:
            copy_reused(model_seed, run_dir)
        except Exception as exc:
            print(f"reuse_failed cell={cell} error={exc}", flush=True)
            failed.append(cell)
        continue
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and complete(run_dir):
        print(f"skip cell={cell}", flush=True)
        cleanup(run_dir)
        continue
    cmd = command(model_seed, prototype_seed, run_dir)
    print(f"run cell={cell}", flush=True)
    started = time.time()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("cmd=" + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nscript_runtime_seconds={time.time() - started:.6f}\n")
        log.write(f"script_exit_code={proc.returncode}\n")
    if proc.returncode != 0:
        tail = "\n".join((run_dir / "train.log").read_text(encoding="utf-8", errors="ignore").splitlines()[-30:])
        (run_dir / "result.json").write_text(json.dumps({
            "status": "failed", "model_seed": model_seed, "prototype_seed": prototype_seed,
            "exit_code": proc.returncode, "error_summary": tail[-2000:],
        }, indent=2, sort_keys=True), encoding="utf-8")
        failed.append(cell)
    cleanup(run_dir)

print(f"run_failures={failed}")
if failed:
    raise SystemExit(1)
PY

RUN_STATUS=$?
"${PYTHON_BIN}" scripts/summarize_l3_seed_source_validation.py "${OUT_DIR}"
SUMMARY_STATUS=$?
printf 'OUT_DIR=%s\n' "${OUT_DIR}"
if [[ ${RUN_STATUS} -ne 0 || ${SUMMARY_STATUS} -ne 0 ]]; then
  exit 1
fi
