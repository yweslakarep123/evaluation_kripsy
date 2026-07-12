#!/usr/bin/env bash
# Sweep FlowPolicy NFE to match DP p3/p4 (~1.0).
#
# Grid: 8 → 10 → 16 → 32 → 64 → 100
# Early-stop when p3>=0.98 and p4>=0.95
# Skips NFE dirs that already have eval_metrics.json
#
# Usage (from anywhere):
#   bash /home/daffa/Documents/experiment/scripts/run_fp_nfe_match_dp.sh
#   DEVICE=cuda:0 bash scripts/run_fp_nfe_match_dp.sh
#   bash scripts/run_fp_nfe_match_dp.sh --dry-run
#   bash scripts/run_fp_nfe_match_dp.sh --analyze-only   # after sweep finishes
#
# Then analyze (also auto-run at end unless --dry-run):
#   python scripts/analyze_fp_nfe_match.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FP_DIR="${EXPERIMENT_ROOT}/kripsy12/FlowPolicy"
CKPT="${FP_DIR}/data/outputs/baseline_42/latest-001.ckpt"
# Prefer checkpoint path recorded by the main Kitchen eval if present
MAIN_METRICS="${FP_DIR}/data/kitchen_eval/flowpolicy/seed_baseline_42/eval_metrics.json"
OUT_ROOT="${FP_DIR}/data/kitchen_eval_nfe/flowpolicy"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/fp_nfe_match_$(date +%Y%m%d_%H%M%S).log"
N_EPISODES="${N_EPISODES:-50}"
DEVICE="${DEVICE:-cuda:0}"
NFE_LIST=(8 10 16 32 64 100)
DRY_RUN=0
ANALYZE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --analyze-only) ANALYZE_ONLY=1 ;;
    --device) DEVICE="$2"; shift ;;
    --n_episodes) N_EPISODES="$2"; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1 (try --help)" >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

resolve_ckpt() {
  if [[ -f "${MAIN_METRICS}" ]]; then
    local from_metrics
    from_metrics="$(python3 -c "import json,sys; p=json.load(open(sys.argv[1])).get('checkpoint',''); print(p or '')" "${MAIN_METRICS}" 2>/dev/null || true)"
    if [[ -n "${from_metrics}" && -f "${from_metrics}" ]]; then
      echo "${from_metrics}"
      return 0
    fi
  fi
  echo "${CKPT}"
}

read_px() {
  local metrics="$1"
  python3 -c "
import json, sys
m=json.load(open(sys.argv[1]))
px=m.get('multistage_metrics',{}).get('all_7_tasks',{}).get('px',{})
lat=m.get('timing_ms',{}).get('inference_latency',{}).get('mean')
eps=m.get('episodes') or []
mt=(sum(e.get('num_tasks_completed',0) for e in eps)/len(eps)) if eps else float('nan')
print(f\"{px.get('p3', float('nan'))} {px.get('p4', float('nan'))} {mt} {lat}\")
" "${metrics}"
}

hit_target() {
  # returns 0 if p3>=0.98 and p4>=0.95
  python3 -c "import sys; p3=float(sys.argv[1]); p4=float(sys.argv[2]); sys.exit(0 if (p3>=0.98 and p4>=0.95) else 1)" "$1" "$2"
}

run_one() {
  local nfe="$1"
  local out_dir="${OUT_ROOT}/seed_baseline_42_nfe${nfe}_sseed0"
  if [[ -f "${out_dir}/eval_metrics.json" ]]; then
    log "SKIP nfe=${nfe} (exists: ${out_dir}/eval_metrics.json)"
    return 0
  fi
  # Remove incomplete dir left by a killed run (no metrics yet)
  if [[ -d "${out_dir}" && ! -f "${out_dir}/eval_metrics.json" ]]; then
    log "WARN: removing incomplete output dir ${out_dir}"
    rm -rf "${out_dir}"
  fi
  log "RUN FlowPolicy nfe=${nfe} episodes=${N_EPISODES}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "  DRY-RUN would write ${out_dir}"
    return 0
  fi
  cd "${FP_DIR}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate flowpolicy-kitchen
  export MUJOCO_GL=egl
  python eval_kitchen.py \
    -c "${CKPT}" \
    --output_root "${OUT_ROOT}" \
    --n_episodes "${N_EPISODES}" \
    --device "${DEVICE}" \
    --num_inference_steps "${nfe}" \
    --sampling_seed 0 \
    --no-video \
    --no-save-trajectory-logs \
    2>&1 | tee -a "${LOG_FILE}"
}

# --- analyze only ---
if [[ "${ANALYZE_ONLY}" -eq 1 ]]; then
  cd "${EXPERIMENT_ROOT}"
  python3 scripts/analyze_fp_nfe_match.py
  exit 0
fi

CKPT="$(resolve_ckpt)"
log "FP NFE match sweep start. log=${LOG_FILE}"
log "ckpt=${CKPT}"
log "episodes=${N_EPISODES} device=${DEVICE} dry_run=${DRY_RUN}"

# Preflight
if [[ ! -f "${CKPT}" ]]; then
  log "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi
if [[ ! -f "${FP_DIR}/data/kitchen/all_init_qpos.npy" ]]; then
  log "ERROR: missing Kitchen dataset at ${FP_DIR}/data/kitchen/all_init_qpos.npy"
  exit 1
fi

HIT_TARGET=0
for nfe in "${NFE_LIST[@]}"; do
  run_one "${nfe}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi
  metrics="${OUT_ROOT}/seed_baseline_42_nfe${nfe}_sseed0/eval_metrics.json"
  if [[ ! -f "${metrics}" ]]; then
    log "ERROR: missing ${metrics} after run"
    exit 1
  fi
  read -r p3 p4 mt lat <<<"$(read_px "${metrics}")"
  log "RESULT nfe=${nfe} p3=${p3} p4=${p4} mean_tasks=${mt} lat_ms=${lat}"

  if hit_target "${p3}" "${p4}"; then
    log "TARGET HIT at nfe=${nfe} (p3>=0.98 and p4>=0.95). Early-stop."
    HIT_TARGET=1
    break
  fi
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "DRY-RUN complete (no eval / no analyze)."
  exit 0
fi

log "Sweep loop done. hit_target=${HIT_TARGET}"
log "Analyzing..."
cd "${EXPERIMENT_ROOT}"
python3 scripts/analyze_fp_nfe_match.py 2>&1 | tee -a "${LOG_FILE}"
log "Done. Report: ${EXPERIMENT_ROOT}/data/kitchen_eval_plots/nfe_variance/fp_nfe_match_report.txt"
