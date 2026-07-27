#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${ROOT_DIR}/logs/direct_node_time_stabilization/${RUN_TIMESTAMP}"

RUN_ARGS=()
if [[ $# -eq 0 ]]; then
  RUN_ARGS+=("--all" "--resume")
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      RUN_ARGS+=("--all")
      shift
      ;;
    --resume)
      RUN_ARGS+=("--resume")
      shift
      ;;
    --dataset)
      RUN_ARGS+=("--dataset" "$2")
      shift 2
      ;;
    --config)
      RUN_ARGS+=("--config" "$2")
      shift 2
      ;;
    --seed)
      RUN_ARGS+=("--seed" "$2")
      shift 2
      ;;
    --phase)
      RUN_ARGS+=("--phase" "$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUT_DIR}/code_info"

{
  git rev-parse HEAD
  git log -1 --oneline
} > "${OUT_DIR}/code_info/commit.txt" 2>&1

{
  echo "which_python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PY
  echo "DEVICE=${DEVICE}"
} > "${OUT_DIR}/code_info/environment.txt" 2>&1

nvidia-smi > "${OUT_DIR}/code_info/gpu_info.txt" 2>&1 || true

cat > "${OUT_DIR}/code_info/common_config.json" <<'JSON'
{
  "method": "ETGC",
  "experiment": "direct_node_time_stabilization",
  "configs": ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10"],
  "seeds": [42, 43, 44],
  "epochs": 30,
  "forest_seed": 20260725,
  "edge_ppr_method": "temporal_state_forest",
  "edge_ppr_topk": -1,
  "cluster_loss_type": "trace_mincut",
  "orth_type": "orth",
  "lambda_edge_ncut": 0.5,
  "lambda_proj": 0.0,
  "lambda_bal": 0.0,
  "cluster_output_bias_mode": "none",
  "cluster_input_norm": "layernorm",
  "cluster_init_mode": "random_orthogonal",
  "require_pretrained_node2vec": 1
}
JSON

COMMAND=(
  "${PYTHON_BIN}"
  "scripts/summarize_direct_node_time_stabilization.py"
  "--run"
  "--prewarm"
  "--out-dir" "${OUT_DIR}"
  "--root" "${ROOT_DIR}"
  "--python" "${PYTHON_BIN}"
  "--device" "${DEVICE}"
)
COMMAND+=("${RUN_ARGS[@]}")

{
  echo "launcher_out_dir=${OUT_DIR}"
  printf 'launcher_command='
  printf '%q ' "${COMMAND[@]}"
  echo
} | tee -a "${OUT_DIR}/launcher.log"

"${COMMAND[@]}" 2>&1 | tee -a "${OUT_DIR}/launcher.log"
EXIT_CODE=${PIPESTATUS[0]}

echo "launcher_exit_code=${EXIT_CODE}" | tee -a "${OUT_DIR}/launcher.log"
exit "${EXIT_CODE}"
