#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="logs/legacy_target_tuning_20260710"
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

LEGACY_COMMON=(
  --edge_ppr_method legacy_temporal_forest
  --edge_ppr_topk 0
  --lambda_prox 1.0
  --lambda_edge_ncut 1.0
  --lambda_bal 50.0
  --eval_every 1
)

# Patent: first reproduce the historical >0.50 ACC setting.
run_case patent_legacy_a10_b3_k10_s10_lr3e4_proj02_e20 patent \
  "${LEGACY_COMMON[@]}" \
  --alpha 0.1 \
  --beta 3.0 \
  --edge_neighbor_k 10 \
  --forest_samples 10 \
  --batch_size 1024 \
  --epoch 20 \
  --learning_rate 0.0003 \
  --lambda_proj 0.2

run_case patent_legacy_a10_b3_k10_s20_lr3e4_proj02_e30 patent \
  "${LEGACY_COMMON[@]}" \
  --alpha 0.1 \
  --beta 3.0 \
  --edge_neighbor_k 10 \
  --forest_samples 20 \
  --batch_size 1024 \
  --epoch 30 \
  --learning_rate 0.0003 \
  --lambda_proj 0.2

run_case patent_legacy_a20_b3_k10_s10_lr3e4_proj02_e20 patent \
  "${LEGACY_COMMON[@]}" \
  --alpha 0.2 \
  --beta 3.0 \
  --edge_neighbor_k 10 \
  --forest_samples 10 \
  --batch_size 1024 \
  --epoch 20 \
  --learning_rate 0.0003 \
  --lambda_proj 0.2

# DBLP: historical strong family, with variants aimed at higher ACC.
run_case dblp_legacy_a20_b3_k10_s5_lr3e4_proj03_e15_seed42 dblp \
  "${LEGACY_COMMON[@]}" \
  --alpha 0.2 \
  --beta 3.0 \
  --edge_neighbor_k 10 \
  --forest_samples 5 \
  --batch_size 1024 \
  --epoch 15 \
  --learning_rate 0.0003 \
  --lambda_proj 0.3 \
  --seed 42

run_case dblp_legacy_a20_b3_k10_s10_lr3e4_proj03_e15_seed42 dblp \
  "${LEGACY_COMMON[@]}" \
  --alpha 0.2 \
  --beta 3.0 \
  --edge_neighbor_k 10 \
  --forest_samples 10 \
  --batch_size 1024 \
  --epoch 15 \
  --learning_rate 0.0003 \
  --lambda_proj 0.3 \
  --seed 42

run_case dblp_legacy_a20_b3_k10_s5_lr5e4_proj03_e20_seed42 dblp \
  "${LEGACY_COMMON[@]}" \
  --alpha 0.2 \
  --beta 3.0 \
  --edge_neighbor_k 10 \
  --forest_samples 5 \
  --batch_size 1024 \
  --epoch 20 \
  --learning_rate 0.0005 \
  --lambda_proj 0.3 \
  --seed 42

run_case dblp_legacy_a30_b3_k10_s5_lr3e4_proj03_e15_seed42 dblp \
  "${LEGACY_COMMON[@]}" \
  --alpha 0.3 \
  --beta 3.0 \
  --edge_neighbor_k 10 \
  --forest_samples 5 \
  --batch_size 1024 \
  --epoch 15 \
  --learning_rate 0.0003 \
  --lambda_proj 0.3 \
  --seed 42

run_case dblp_legacy_a20_b1_k10_s5_lr3e4_proj03_e15_seed42 dblp \
  "${LEGACY_COMMON[@]}" \
  --alpha 0.2 \
  --beta 1.0 \
  --edge_neighbor_k 10 \
  --forest_samples 5 \
  --batch_size 1024 \
  --epoch 15 \
  --learning_rate 0.0003 \
  --lambda_proj 0.3 \
  --seed 42

run_case dblp_legacy_a20_b3_k20_s5_lr3e4_proj03_e15_seed42 dblp \
  "${LEGACY_COMMON[@]}" \
  --alpha 0.2 \
  --beta 3.0 \
  --edge_neighbor_k 20 \
  --forest_samples 5 \
  --batch_size 1024 \
  --epoch 15 \
  --learning_rate 0.0003 \
  --lambda_proj 0.3 \
  --seed 42

for seed in 1 7 123; do
  run_case "dblp_legacy_a20_b3_k10_s5_lr3e4_proj03_e10_seed${seed}" dblp \
    "${LEGACY_COMMON[@]}" \
    --alpha 0.2 \
    --beta 3.0 \
    --edge_neighbor_k 10 \
    --forest_samples 5 \
    --batch_size 1024 \
    --epoch 10 \
    --learning_rate 0.0003 \
    --lambda_proj 0.3 \
    --seed "${seed}"
done

echo "summary: ${SUMMARY}"
