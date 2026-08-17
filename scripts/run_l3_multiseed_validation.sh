#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "No python or python3 executable found." >&2
  exit 127
fi

if [[ -z "${LOSS_VALIDATION_DIR:-}" ]]; then
  echo "LOSS_VALIDATION_DIR must point to the completed phase-2 result directory." >&2
  exit 2
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/l3_multiseed_validation/${RUN_TIMESTAMP}}"
mkdir -p "${OUT_DIR}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TEMPORAL_FOREST_WORKERS="${TEMPORAL_FOREST_WORKERS:-1}"
export TEMPORAL_FOREST_CHUNK_SAMPLES="${TEMPORAL_FOREST_CHUNK_SAMPLES:-1}"
export TEMPORAL_FOREST_COMBINE_CHUNKS="${TEMPORAL_FOREST_COMBINE_CHUNKS:-8}"

"${PYTHON_BIN}" scripts/run_l3_multiseed_validation.py \
  --root-dir "${ROOT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --reference-dir "${LOSS_VALIDATION_DIR}" \
  --asset-root "${ASSET_ROOT:-${ROOT_DIR}}" \
  --python-bin "${PYTHON_BIN}" \
  --device "${DEVICE:-cuda:0}" \
  "$@"

STATUS=$?
printf 'OUT_DIR=%s\n' "${OUT_DIR}"
exit ${STATUS}
