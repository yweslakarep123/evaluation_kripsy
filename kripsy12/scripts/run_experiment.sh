#!/usr/bin/env bash
# Eksperimen penuh: (1) baseline lalu (2) Hyperband (Li et al., 2018) +
# (3) rerun pemenang top-1 di full seeds × profiles.
#
# Default Hyperband: R=3000, eta=3, s_min=0 (semua bracket sesuai paper).
# Untuk fit ≤ 2 hari, tambahkan `--hyperband-s-min 2` (single-bracket SHA).
#
# Dari akar repositori: ./scripts/run_experiment.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATASET_DIR="${DATASET_DIR:-data/kitchen/kitchen_demos_multitask}"
exec python3 scripts/run_experiment.py \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --dataset-dir "$DATASET_DIR" \
  --hyperband-max-epochs 3000 \
  --hyperband-eta 3 \
  --hyperband-s-min 0 \
  --hyperband-seed 99 \
  --hyperband-search-train-seed 0 \
  --hyperband-search-profile standard \
  --cv-seed 12345 \
  "$@"
