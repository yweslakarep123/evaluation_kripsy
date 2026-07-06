#!/usr/bin/env bash
# Hanya Hyperband (Li et al., 2018) + re-run pemenang top-1 di full seeds × profiles.
# Tanpa baseline.
#
# Default: R=3000 (= baseline num_epochs), eta=3 (sesuai paper).
# Hyperband berjalan di 1 seed × 1 profile (default seed=0, profile=standard);
# pemenang top-1 di-rerun di --seeds × --profiles (default 3 × 2 = 6 run).
#
# Anggaran waktu (perkiraan; lihat README untuk detail):
#   --hyperband-s-min 0 (default): SEMUA bracket s=s_max..0  → paling robust,
#       paling mahal (bisa > 2 hari untuk R=3000 tergantung kecepatan training).
#   --hyperband-s-min 2: single-bracket SHA s=2 saja             → tercepat,
#       sesuai paper "bracket s=k as a standalone SuccessiveHalving".
#
# Dari akar repositori: ./scripts/run_hyperband_only.sh
# Argumen tambahan diteruskan ke run_experiment.py.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET_DIR="${DATASET_DIR:-data/kitchen/kitchen_demos_multitask}"
exec python3 scripts/run_experiment.py \
  --hyperband-only \
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
