#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON:-python}"
DEVICE_ARG="${DEVICE:-cuda:0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/kmeans_contribution_orthqa_overnight/${RUN_TIMESTAMP}}"
CONFIG_SOURCE_DIR="${CONFIG_SOURCE_DIR:-${ROOT_DIR}/logs/trace_mincut_global/20260724_002532/phase1_diagnosis}"

GROUPS=("all")
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME=1
      shift
      ;;
    --all)
      GROUPS=("all")
      shift
      ;;
    --group)
      if [[ $# -lt 2 ]]; then
        echo "--group requires A, B, C, or D" >&2
        exit 2
      fi
      GROUPS=("$2")
      shift 2
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
  env | sort | grep -E '^(DEVICE|CUDA_VISIBLE_DEVICES|OPENBLAS_NUM_THREADS|OMP_NUM_THREADS|TEMPORAL_FOREST_)=' || true
} > "${OUT_DIR}/code_info/environment.txt"

{
  nvidia-smi || true
} > "${OUT_DIR}/code_info/gpu_info.txt"

cat > "${OUT_DIR}/code_info/common_config.json" <<EOF
{
  "method": "ETGC",
  "forest_seed": 20260725,
  "groups": ["${GROUPS[*]}"],
  "epochs": 30,
  "edge_ppr_method": "temporal_state_forest",
  "edge_ppr_topk": -1,
  "cluster_loss_type": "trace_mincut",
  "lambda_edge_ncut": 0.5,
  "lambda_proj": 0.0,
  "lambda_bal": 0.0,
  "node_emb_mode": "frozen"
}
EOF

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TEMPORAL_FOREST_WORKERS="${TEMPORAL_FOREST_WORKERS:-1}"
export TEMPORAL_FOREST_CHUNK_SAMPLES="${TEMPORAL_FOREST_CHUNK_SAMPLES:-1}"
export TEMPORAL_FOREST_COMBINE_CHUNKS="${TEMPORAL_FOREST_COMBINE_CHUNKS:-8}"

CMD=(
  "${PYTHON_BIN}" "scripts/summarize_kmeans_orthqa_runs.py"
  "--run"
  "--prewarm"
  "--out-dir" "${OUT_DIR}"
  "--root" "${ROOT_DIR}"
  "--python" "${PYTHON_BIN}"
  "--device" "${DEVICE_ARG}"
  "--config-source-dir" "${CONFIG_SOURCE_DIR}"
  "--groups" "${GROUPS[@]}"
)
if [[ "${RESUME}" -eq 1 ]]; then
  CMD+=("--resume")
fi

printf 'launcher_out_dir=%s\n' "${OUT_DIR}" | tee "${OUT_DIR}/launcher.log"
printf 'launcher_command=' | tee -a "${OUT_DIR}/launcher.log"
printf '%q ' "${CMD[@]}" | tee -a "${OUT_DIR}/launcher.log"
printf '\n' | tee -a "${OUT_DIR}/launcher.log"

"${CMD[@]}" 2>&1 | tee -a "${OUT_DIR}/launcher.log"
exit "${PIPESTATUS[0]}"
