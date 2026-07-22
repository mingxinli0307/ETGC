#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="logs/cuda_large_samples500_20260709"
SUMMARY="${OUT_DIR}/summary.tsv"
DEVICE="${DEVICE:-cuda:0}"
export TEMPORAL_FOREST_WORKERS="${TEMPORAL_FOREST_WORKERS:-8}"
export TEMPORAL_FOREST_CHUNK_SAMPLES="${TEMPORAL_FOREST_CHUNK_SAMPLES:-1}"
export TEMPORAL_FOREST_COMBINE_CHUNKS="${TEMPORAL_FOREST_COMBINE_CHUNKS:-8}"

mkdir -p "${OUT_DIR}"
if [[ ! -s "${SUMMARY}" || "${RESET_SUMMARY:-0}" == "1" ]]; then
  printf "dataset\tbest_epoch\tACC\tNMI\tARI\tF1\tlog\n" > "${SUMMARY}"
fi

summary_has_dataset() {
  local dataset="$1"
  awk -F '\t' -v dataset="${dataset}" '$1 == dataset { found = 1 } END { exit(found ? 0 : 1) }' "${SUMMARY}"
}

record_result() {
  local dataset="$1"
  local log="$2"
  python - "$dataset" "$log" "$SUMMARY" <<'PY'
import re
import sys
from pathlib import Path

dataset, log_path, summary_path = sys.argv[1:]
text = Path(log_path).read_text(errors="ignore")
best_epoch = re.search(r"^best_epoch=(\d+)", text, re.M)
metrics = {k: re.search(rf"^{k}=([0-9.]+)", text, re.M) for k in ["ACC", "NMI", "ARI", "F1"]}
if best_epoch is None or any(v is None for v in metrics.values()):
    raise SystemExit(f"Could not parse final metrics from {log_path}")
row = [
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
  local dataset="$1"
  local log="$2"
  shift 2
  echo "[$(date '+%F %T')] start ${dataset}"
  python -u edge_main.py --dataset "${dataset}" --device "${DEVICE}" "$@" > "${log}" 2>&1
  record_result "${dataset}" "${log}"
  echo "[$(date '+%F %T')] done ${dataset}"
}

COMMON_ARGS=(
  --edge_ppr_method temporal_state_forest
  --alpha 0.2
  --beta 5.0
  --edge_neighbor_k 3
  --edge_ppr_topk 20
  --forest_samples 500
  --learning_rate 0.0001
  --lambda_prox 1.0
  --lambda_edge_ncut 0.5
  --lambda_proj 0.3
  --lambda_bal 50.0
  --batch_size 4096
  --epoch 20
  --eval_every 5
)

for dataset in arXivAI arXivMath arXivCS brain; do
  if summary_has_dataset "${dataset}"; then
    echo "[$(date '+%F %T')] skip ${dataset}; already in ${SUMMARY}"
    continue
  fi
  run_case "${dataset}" "${OUT_DIR}/${dataset}_cuda_samples500.log" "${COMMON_ARGS[@]}"
done

echo "summary: ${SUMMARY}"
