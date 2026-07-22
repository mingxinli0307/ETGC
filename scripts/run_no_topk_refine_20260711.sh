#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="logs/no_topk_refine_20260711"
SUMMARY="${OUT_DIR}/summary.tsv"
DEVICE="${DEVICE:-cuda:0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TEMPORAL_FOREST_WORKERS="${TEMPORAL_FOREST_WORKERS:-8}"
export TEMPORAL_FOREST_CHUNK_SAMPLES="${TEMPORAL_FOREST_CHUNK_SAMPLES:-1}"
export TEMPORAL_FOREST_COMBINE_CHUNKS="${TEMPORAL_FOREST_COMBINE_CHUNKS:-8}"

mkdir -p "${OUT_DIR}"
if [[ ! -s "${SUMMARY}" || "${RESET_SUMMARY:-0}" == "1" ]]; then
  printf "case\tdataset\tbest_epoch\tACC\tNMI\tARI\tF1\tlog\n" > "${SUMMARY}"
fi

summary_has_case() {
  local case_name="$1"
  awk -F '\t' -v case_name="${case_name}" '$1 == case_name { found = 1 } END { exit(found ? 0 : 1) }' "${SUMMARY}"
}

record_result() {
  local case_name="$1"
  local dataset="$2"
  local log="$3"
  python - "$case_name" "$dataset" "$log" "$SUMMARY" <<'PY'
import re
import sys
from pathlib import Path

case_name, dataset, log_path, summary_path = sys.argv[1:]
text = Path(log_path).read_text(errors="ignore")
best_epoch = re.search(r"^best_epoch=(\d+)", text, re.M)
metrics = {k: re.search(rf"^{k}=([0-9.]+)", text, re.M) for k in ["ACC", "NMI", "ARI", "F1"]}
if best_epoch is None or any(v is None for v in metrics.values()):
    raise SystemExit(f"Could not parse final metrics from {log_path}")
row = [
    case_name,
    dataset,
    best_epoch.group(1),
    metrics["ACC"].group(1),
    metrics["NMI"].group(1),
    metrics["ARI"].group(1),
    metrics["F1"].group(1),
    log_path,
]
with Path(summary_path).open("a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
print("\t".join(row), flush=True)
PY
}

run_case() {
  local case_name="$1"
  local dataset="$2"
  local log="${OUT_DIR}/${case_name}.log"
  shift 2
  if summary_has_case "${case_name}"; then
    echo "[$(date '+%F %T')] skip ${case_name}; already in ${SUMMARY}"
    return
  fi
  echo "[$(date '+%F %T')] start ${case_name}"
  python -u edge_main.py --dataset "${dataset}" --device "${DEVICE}" "$@" > "${log}" 2>&1
  record_result "${case_name}" "${dataset}" "${log}"
  echo "[$(date '+%F %T')] done ${case_name}"
}

COMMON=(
  --edge_ppr_method temporal_state_forest
  --edge_neighbor_k 3
  --edge_ppr_topk 0
  --learning_rate 0.0001
  --lambda_prox 1.0
  --lambda_edge_ncut 0.5
  --lambda_proj 0.3
  --lambda_bal 50.0
  --batch_size 4096
  --epoch 5
  --eval_every 1
  --seed 42
)

run_case arxivai_state_s20_topk0_a20_b5_e5 arXivAI \
  "${COMMON[@]}" --alpha 0.2 --beta 5.0 --forest_samples 20

run_case arxivai_state_s10_topk0_a10_b5_e5 arXivAI \
  "${COMMON[@]}" --alpha 0.1 --beta 5.0 --forest_samples 10

run_case arxivai_state_s10_topk0_a20_b3_e5 arXivAI \
  "${COMMON[@]}" --alpha 0.2 --beta 3.0 --forest_samples 10

run_case arxivai_state_s10_topk0_a20_b5_lr3e4_b2048_e5 arXivAI \
  "${COMMON[@]}" --alpha 0.2 --beta 5.0 --forest_samples 10 \
  --learning_rate 0.0003 --batch_size 2048

run_case brain_state_s20_topk0_a20_b5_e5 brain \
  "${COMMON[@]}" --alpha 0.2 --beta 5.0 --forest_samples 20

run_case brain_state_s10_topk0_a10_b5_e5 brain \
  "${COMMON[@]}" --alpha 0.1 --beta 5.0 --forest_samples 10

echo "summary: ${SUMMARY}"
