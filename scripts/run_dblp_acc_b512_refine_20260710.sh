#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="logs/dblp_acc_b512_refine_20260710"
SUMMARY="${OUT_DIR}/summary.tsv"
DEVICE="${DEVICE:-cuda:0}"

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
  local log="${OUT_DIR}/${case_name}.log"
  shift
  if summary_has_case "${case_name}"; then
    echo "[$(date '+%F %T')] skip ${case_name}; already in ${SUMMARY}"
    return
  fi
  echo "[$(date '+%F %T')] start ${case_name}"
  python -u edge_main.py --dataset dblp --device "${DEVICE}" "$@" > "${log}" 2>&1
  record_result "${case_name}" "dblp" "${log}"
  echo "[$(date '+%F %T')] done ${case_name}"
}

COMMON=(
  --edge_ppr_method legacy_temporal_forest
  --alpha 0.2
  --beta 3.0
  --edge_neighbor_k 10
  --edge_ppr_topk 0
  --forest_samples 10
  --eval_every 1
  --seed 42
  --batch_size 512
)

run_case b512_lr1e4_ncut1_proj03_bal50_e8 \
  "${COMMON[@]}" --epoch 8 --learning_rate 0.0001 \
  --lambda_prox 1.0 --lambda_edge_ncut 1.0 --lambda_proj 0.3 --lambda_bal 50.0

run_case b512_lr5e5_ncut1_proj03_bal50_e8 \
  "${COMMON[@]}" --epoch 8 --learning_rate 0.00005 \
  --lambda_prox 1.0 --lambda_edge_ncut 1.0 --lambda_proj 0.3 --lambda_bal 50.0

run_case b512_lr3e4_ncut05_proj03_bal50_e8 \
  "${COMMON[@]}" --epoch 8 --learning_rate 0.0003 \
  --lambda_prox 1.0 --lambda_edge_ncut 0.5 --lambda_proj 0.3 --lambda_bal 50.0

run_case b512_lr1e4_ncut05_proj03_bal50_e8 \
  "${COMMON[@]}" --epoch 8 --learning_rate 0.0001 \
  --lambda_prox 1.0 --lambda_edge_ncut 0.5 --lambda_proj 0.3 --lambda_bal 50.0

run_case b512_lr3e4_ncut1_proj02_bal50_e8 \
  "${COMMON[@]}" --epoch 8 --learning_rate 0.0003 \
  --lambda_prox 1.0 --lambda_edge_ncut 1.0 --lambda_proj 0.2 --lambda_bal 50.0

run_case b512_lr1e4_ncut1_proj02_bal50_e8 \
  "${COMMON[@]}" --epoch 8 --learning_rate 0.0001 \
  --lambda_prox 1.0 --lambda_edge_ncut 1.0 --lambda_proj 0.2 --lambda_bal 50.0

run_case b512_lr3e4_ncut05_proj02_bal50_e8 \
  "${COMMON[@]}" --epoch 8 --learning_rate 0.0003 \
  --lambda_prox 1.0 --lambda_edge_ncut 0.5 --lambda_proj 0.2 --lambda_bal 50.0

run_case b512_lr3e4_ncut1_proj03_bal75_e8 \
  "${COMMON[@]}" --epoch 8 --learning_rate 0.0003 \
  --lambda_prox 1.0 --lambda_edge_ncut 1.0 --lambda_proj 0.3 --lambda_bal 75.0

run_case b512_a20_b2_k10_s10_lr3e4_ncut1_proj03_bal50_e8 \
  "${COMMON[@]}" --beta 2.0 --epoch 8 --learning_rate 0.0003 \
  --lambda_prox 1.0 --lambda_edge_ncut 1.0 --lambda_proj 0.3 --lambda_bal 50.0

run_case b512_a20_b3_k5_s10_lr3e4_ncut1_proj03_bal50_e8 \
  "${COMMON[@]}" --edge_neighbor_k 5 --epoch 8 --learning_rate 0.0003 \
  --lambda_prox 1.0 --lambda_edge_ncut 1.0 --lambda_proj 0.3 --lambda_bal 50.0

run_case b512_a20_b3_k10_s20_lr3e4_ncut1_proj03_bal50_e8 \
  "${COMMON[@]}" --forest_samples 20 --epoch 8 --learning_rate 0.0003 \
  --lambda_prox 1.0 --lambda_edge_ncut 1.0 --lambda_proj 0.3 --lambda_bal 50.0

echo "summary: ${SUMMARY}"
