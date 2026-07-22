#!/usr/bin/env bash
set -euo pipefail

DATASETS=(school hospital workplace primary highschool)

for dataset in "${DATASETS[@]}"; do
  python edge_main.py \
    --dataset "${dataset}" \
    --device cuda \
    --edge_ppr_method forest \
    --forest_samples 20 \
    --edge_neighbor_k 10 \
    --edge_ppr_topk 20
done
