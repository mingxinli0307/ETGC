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

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/forest500_full_validation/${RUN_TIMESTAMP}}"
DEVICE_ARG="${DEVICE:-cuda:0}"
FOREST_SAMPLES_ARG="${FOREST_SAMPLES:-500}"
RESUME_ARG="--resume"

if [[ "${FOREST_SAMPLES_ARG}" != "500" ]]; then
  echo "This launcher requires FOREST_SAMPLES=500; got ${FOREST_SAMPLES_ARG}." >&2
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-resume)
      RESUME_ARG="--no-resume"
      shift
      ;;
    --resume)
      RESUME_ARG="--resume"
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
  echo "device=${DEVICE_ARG}"
  echo "forest_samples=${FOREST_SAMPLES_ARG}"
  echo "which_python=$(command -v "${PYTHON_BIN}" || true)"
  "${PYTHON_BIN}" --version 2>&1 || true
  "${PYTHON_BIN}" - <<'PY' || true
import torch
print("torch_version=" + str(torch.__version__))
print("cuda_available=" + str(torch.cuda.is_available()))
print("cuda_device=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
PY
  env | sort | grep -E '^(DEVICE|FOREST_SAMPLES|CUDA_VISIBLE_DEVICES|OPENBLAS_NUM_THREADS|OMP_NUM_THREADS|TEMPORAL_FOREST_)=' || true
} > "${OUT_DIR}/code_info/environment.txt"

"${PYTHON_BIN}" - "${OUT_DIR}/code_info/common_config.json" "${ASSET_ROOT:-${ROOT_DIR}}" <<'PY'
import json
import sys

path, asset_root = sys.argv[1:]
common = {
    "method": "ETGC",
    "purpose": "forest_samples=500 full validation",
    "forest_samples": 500,
    "forest_seed": 20260725,
    "edge_neighbor_k": -1,
    "edge_ppr_topk": -1,
    "datasets": ["school", "dblp", "patent", "arXivAI"],
    "seeds": [42, 43],
    "rank1_scope": "School C0-C7, 3 epochs, 16 runs",
    "loss_scope": "four datasets x L1-L4 x two seeds, 20 epochs, 32 runs",
    "total_runs": 48,
    "asset_root": asset_root,
}
with open(path, "w", encoding="utf-8") as writer:
    json.dump(common, writer, indent=2, sort_keys=True)
PY

failures=()

run_child() {
  local label="$1"
  shift
  echo "start ${label}"
  if "$@"; then
    echo "complete ${label}"
  else
    local status=$?
    echo "failed ${label} exit_code=${status}" >&2
    failures+=("${label}:${status}")
  fi
}

run_child "rank1_school" \
  env PYTHON="${PYTHON_BIN}" DEVICE="${DEVICE_ARG}" FOREST_SAMPLES=500 \
  ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}}" \
  CONFIG_SOURCE_DIR="${ASSET_ROOT:-${ROOT_DIR}}/logs/trace_mincut_global/20260724_002532/phase1_diagnosis" \
  OUT_DIR="${OUT_DIR}/rank1_cause_validation" \
  bash scripts/run_rank1_cause_validation.sh "${RESUME_ARG}"

for dataset in school dblp patent arXivAI; do
  run_child "loss_${dataset}" \
    env PYTHON="${PYTHON_BIN}" DEVICE="${DEVICE_ARG}" DATASET="${dataset}" FOREST_SAMPLES=500 \
    ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}}" \
    C6_REFERENCE_DIR= OUT_DIR="${OUT_DIR}/loss_formulation_validation/${dataset}" \
    bash scripts/run_loss_formulation_validation.sh "${RESUME_ARG}"
done

"${PYTHON_BIN}" - "${OUT_DIR}/run_status.json" "${failures[@]}" <<'PY'
import json
import sys

path = sys.argv[1]
failures = sys.argv[2:]
with open(path, "w", encoding="utf-8") as writer:
    json.dump({"status": "success" if not failures else "partial_failure", "failures": failures}, writer, indent=2)
PY

printf 'OUT_DIR=%s\n' "${OUT_DIR}"
printf 'failures=%s\n' "${failures[*]:-none}"
[[ ${#failures[@]} -eq 0 ]]
