#!/usr/bin/env bash
# Hanya 3 pelatihan baseline: 3 seed × 1 profil (standard).
# Split demo MJL: 70%% train / 20%% val / 10%% test (~605 episode).
# Eval policy: simulasi MuJoCo KitchenAllV0, multi-seed 0,42,101 via infer_kitchen_lowdim.py.
# Dari akar repositori: ./scripts/run_baseline_only.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATASET_DIR="${DATASET_DIR:-data/kitchen/kitchen_demos_multitask}"
exec python3 scripts/run_experiment.py \
  --baseline-only \
  --seeds 0 42 101 \
  --profiles standard \
  --dataset-dir "$DATASET_DIR" \
  --cv-seed 12345 \
  --n-infer-episodes 50 \
  "$@"
