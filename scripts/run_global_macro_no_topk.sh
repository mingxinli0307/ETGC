#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

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

export TEMPORAL_FOREST_WORKERS="${TEMPORAL_FOREST_WORKERS:-8}"
export TEMPORAL_FOREST_CHUNK_SAMPLES="${TEMPORAL_FOREST_CHUNK_SAMPLES:-1}"
export TEMPORAL_FOREST_COMBINE_CHUNKS="${TEMPORAL_FOREST_COMBINE_CHUNKS:-8}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="logs/global_ncut_macroF1_no_topk/${RUN_TIMESTAMP}"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
SUMMARY_JSON="${OUT_DIR}/summary.json"
RUN_CONFIG_JSON="${OUT_DIR}/run_config.json"
GIT_DIFF_PATCH="${OUT_DIR}/git_diff.patch"

mkdir -p "${OUT_DIR}"

cat > "${SUMMARY_CSV}" <<'EOF'
dataset,status,seed,num_nodes,num_events,K,ncut_scope,edge_ppr_method,edge_neighbor_k,edge_ppr_topk,Pi_nnz,W_nnz,best_epoch,ACC,NMI,ARI,Macro_F1,runtime_seconds,peak_gpu_memory_mb,log_path,error_summary
EOF

"${PYTHON_BIN}" - "${RUN_CONFIG_JSON}" "${ROOT_DIR}" "${RUN_TIMESTAMP}" <<'PY'
import json
import sys

path, root_dir, run_timestamp = sys.argv[1:]
cfg = {
    "root_dir": root_dir,
    "run_timestamp": run_timestamp,
    "datasets_requested": ["school", "patent", "dblp", "arxivAI", "arxivCS"],
    "seed": 42,
    "edge_ppr_method": "temporal_state_forest",
    "edge_neighbor_k": -1,
    "edge_ppr_topk": -1,
    "ncut_scope": "global",
    "global_q_chunk_size_initial": 8192,
    "global_q_chunk_size_retry_order": [8192, 4096, 2048, 1024],
    "global_ncut_row_block_size_initial": 65536,
    "global_ncut_row_block_size_retry_order_after_q_chunks": [65536, 32768, 16384, 8192, 4096],
    "quiet": 1,
    "temporal_forest_workers": int(__import__("os").environ.get("TEMPORAL_FOREST_WORKERS", "8")),
    "temporal_forest_chunk_samples": int(__import__("os").environ.get("TEMPORAL_FOREST_CHUNK_SAMPLES", "1")),
    "temporal_forest_combine_chunks": int(__import__("os").environ.get("TEMPORAL_FOREST_COMBINE_CHUNKS", "8")),
    "unchanged_defaults": [
        "alpha",
        "beta",
        "forest_samples",
        "learning_rate",
        "epoch",
        "batch_size",
        "lambda_prox",
        "lambda_edge_ncut",
        "lambda_proj",
        "lambda_bal",
        "model dimensions",
    ],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff > "${GIT_DIFF_PATCH}"
else
  printf 'not a git repository\n' > "${GIT_DIFF_PATCH}"
fi

resolve_dataset() {
  local alias="$1"
  local d name
  if [[ -d "dataset/${alias}" && -f "dataset/${alias}/${alias}.txt" && -f "dataset/${alias}/node2label.txt" ]]; then
    printf '%s\n' "${alias}"
    return 0
  fi
  for d in dataset/*; do
    [[ -d "${d}" ]] || continue
    name="$(basename "${d}")"
    if [[ "${name,,}" == "${alias,,}" && -f "${d}/${name}.txt" && -f "${d}/node2label.txt" ]]; then
      printf '%s\n' "${name}"
      return 0
    fi
  done
  return 1
}

write_row_from_log() {
  local dataset="$1"
  local status="$2"
  local log_path="$3"
  local error_summary="$4"
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${dataset}" "${status}" "${log_path}" "${error_summary}" <<'PY'
import csv
import re
import sys
from pathlib import Path

summary_path, dataset, status, log_path, error_summary = sys.argv[1:]
text = Path(log_path).read_text(errors="ignore") if Path(log_path).exists() else ""

def find(pattern, default=""):
    m = re.search(pattern, text, re.M)
    return m.group(1) if m else default

row = {
    "dataset": dataset,
    "status": status,
    "seed": find(r"^seed=(\d+)", "42"),
    "num_nodes": find(r"num_nodes=(\d+)"),
    "num_events": find(r"num_events=(\d+)"),
    "K": find(r"\bK=(\d+)"),
    "ncut_scope": find(r"^ncut_scope=([A-Za-z0-9_+-]+)", "global"),
    "edge_ppr_method": find(r"^edge_ppr_method=([A-Za-z0-9_+-]+)", "temporal_state_forest"),
    "edge_neighbor_k": find(r"edge_neighbor_k=(-?\d+)", "-1"),
    "edge_ppr_topk": find(r"^edge_ppr_topk=(-?\d+)", "-1"),
    "Pi_nnz": find(r"^Pi_E nnz=(\d+)"),
    "W_nnz": find(r"^W_E nnz=(\d+)"),
    "best_epoch": find(r"^best_epoch=(-?\d+)"),
    "ACC": find(r"^ACC=([0-9.]+)"),
    "NMI": find(r"^NMI=([0-9.]+)"),
    "ARI": find(r"^ARI=([0-9.]+)"),
    "Macro_F1": find(r"^Macro_F1=([0-9.]+)"),
    "runtime_seconds": find(r"^runtime_seconds=([0-9.]+)"),
    "peak_gpu_memory_mb": find(r"^peak_gpu_memory_mb=([0-9.]+)"),
    "log_path": str(Path(log_path).resolve()),
    "error_summary": error_summary,
}
with open(summary_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
PY
}

contains_oom() {
  local path="$1"
  grep -Eiq 'out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA out of memory' "${path}"
}

cleanup_cuda_cache() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass
PY
}

append_attempt_log() {
  local attempt_log="$1"
  local final_log="$2"
  local save_full="$3"
  if [[ "${save_full}" == "1" ]]; then
    cat "${attempt_log}" >> "${final_log}"
  else
    "${PYTHON_BIN}" - "${attempt_log}" "${final_log}" <<'PY'
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:])
lines = src.read_text(errors="ignore").splitlines()
kept = []
in_traceback = False
for line in lines:
    if line.startswith("Traceback (most recent call last):"):
        in_traceback = True
        kept.append("[traceback omitted for retry; see earlier attempt]")
        continue
    if in_traceback and (line.startswith(" ") or line.startswith("  File ") or line.startswith("    ")):
        continue
    in_traceback = False
    kept.append(line)
with dst.open("a", encoding="utf-8") as f:
    f.write("\n".join(kept[-120:]))
    f.write("\n")
PY
  fi
}

run_dataset() {
  local alias="$1"
  local dataset="$2"
  local log_path="${OUT_DIR}/${alias}.log"
  local status="failed"
  local error_summary=""
  local traceback_saved=0
  local attempt_log rc q b
  local -a attempts=(
    "8192 65536"
    "4096 65536"
    "2048 65536"
    "1024 65536"
    "1024 32768"
    "1024 16384"
    "1024 8192"
    "1024 4096"
  )

  : > "${log_path}"
  for attempt in "${attempts[@]}"; do
    q="${attempt%% *}"
    b="${attempt##* }"
    attempt_log="${OUT_DIR}/.${alias}_q${q}_b${b}.attempt.log"
    {
      printf 'dataset=%s alias=%s\n' "${dataset}" "${alias}"
      printf 'attempt_global_q_chunk_size=%s attempt_global_ncut_row_block_size=%s\n' "${q}" "${b}"
    } >> "${log_path}"

    "${PYTHON_BIN}" -u edge_main.py \
      --dataset "${dataset}" \
      --edge_ppr_method temporal_state_forest \
      --edge_neighbor_k -1 \
      --edge_ppr_topk -1 \
      --ncut_scope global \
      --global_q_chunk_size "${q}" \
      --global_ncut_row_block_size "${b}" \
      --seed 42 \
      --quiet 1 > "${attempt_log}" 2>&1
    rc=$?

    if [[ "${traceback_saved}" == "0" ]]; then
      append_attempt_log "${attempt_log}" "${log_path}" 1
      traceback_saved=1
    else
      append_attempt_log "${attempt_log}" "${log_path}" 0
    fi

    if [[ "${rc}" == "0" ]]; then
      status="success"
      error_summary=""
      break
    fi

    if contains_oom "${attempt_log}"; then
      error_summary="OOM at global_q_chunk_size=${q}, global_ncut_row_block_size=${b}"
      printf 'retry_after_oom=1\n' >> "${log_path}"
      cleanup_cuda_cache
      continue
    fi

    error_summary="$(tail -n 1 "${attempt_log}" | tr ',' ';' | cut -c1-200)"
    break
  done

  if [[ "${status}" != "success" && -z "${error_summary}" ]]; then
    error_summary="all retry attempts failed"
  fi
  printf 'dataset_status=%s\n' "${status}" >> "${log_path}"
  write_row_from_log "${dataset}" "${status}" "${log_path}" "${error_summary}"
  rm -f "${OUT_DIR}/.${alias}_"*.attempt.log
}

TARGET_ALIASES=(school patent dblp arxivAI arxivCS)
for alias in "${TARGET_ALIASES[@]}"; do
  if dataset="$(resolve_dataset "${alias}")"; then
    run_dataset "${alias}" "${dataset}"
  else
    log_path="${OUT_DIR}/${alias}.log"
    printf 'dataset=%s\nstatus=missing\n' "${alias}" > "${log_path}"
    write_row_from_log "${alias}" "missing" "${log_path}" "dataset directory or required files missing"
  fi
done

"${PYTHON_BIN}" - "${SUMMARY_CSV}" "${SUMMARY_JSON}" <<'PY'
import csv
import json
import sys

csv_path, json_path = sys.argv[1:]
with open(csv_path, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
PY

printf 'OUT_DIR=%s\n' "$(cd "${OUT_DIR}" && pwd)"
printf 'SUMMARY_CSV=%s\n' "$(cd "$(dirname "${SUMMARY_CSV}")" && pwd)/$(basename "${SUMMARY_CSV}")"
