#!/usr/bin/env bash
# Full Kitchen NFE eval: 100 episodes × video + trajectory logs.
#
# Models:
#   DP-CNN / DP-Transformer: train0, train1, train2
#   FlowPolicy:              baseline_42, baseline_43, baseline_44
# NFE grid: 1, 8, 32, 100
# Total: 3 models × 3 ckpts × 4 NFE = 36 runs (3600 episodes)
#
# Usage:
#   bash scripts/run_kitchen_nfe100_full.sh --dry-run
#   bash scripts/run_kitchen_nfe100_full.sh
#   bash scripts/run_kitchen_nfe100_full.sh --model flowpolicy
#   bash scripts/run_kitchen_nfe100_full.sh --model cnn --nfe 8
#   bash scripts/run_kitchen_nfe100_full.sh --analyze-only
#   DEVICE=cuda:0 bash scripts/run_kitchen_nfe100_full.sh --overwrite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DP_DIR="${EXPERIMENT_ROOT}/diffusion_policy"
FP_DIR="${EXPERIMENT_ROOT}/kripsy12/FlowPolicy"
OUT_DP="${DP_DIR}/data/kitchen_eval_nfe100"
OUT_FP="${FP_DIR}/data/kitchen_eval_nfe100/flowpolicy"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/kitchen_nfe100_$(date +%Y%m%d_%H%M%S).log"

N_EPISODES="${N_EPISODES:-100}"
DEVICE="${DEVICE:-cuda:0}"
SAMPLING_SEED=0
NFE_LIST=(1 8 32 100)
MODEL_FILTER="all"
NFE_FILTER=""
DRY_RUN=0
ANALYZE_ONLY=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --analyze-only) ANALYZE_ONLY=1 ;;
    --overwrite) OVERWRITE=1 ;;
    --device) DEVICE="$2"; shift ;;
    --n_episodes) N_EPISODES="$2"; shift ;;
    --model)
      MODEL_FILTER="$2"
      shift
      case "${MODEL_FILTER}" in
        cnn|transformer|flowpolicy|all) ;;
        *)
          echo "Unknown --model ${MODEL_FILTER} (cnn|transformer|flowpolicy|all)" >&2
          exit 1
          ;;
      esac
      ;;
    --nfe) NFE_FILTER="$2"; shift ;;
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

mkdir -p "${LOG_DIR}" "${OUT_DP}" "${OUT_FP}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

ckpt_from_metrics() {
  local metrics="$1"
  python3 -c "
import json, sys
p=json.load(open(sys.argv[1])).get('checkpoint','')
print(p if p else '')
" "${metrics}" 2>/dev/null || true
}

resolve_dp_ckpt() {
  local model="$1"   # diffusion_policy_cnn | diffusion_policy_transformer
  local seed="$2"    # train0 | train1 | train2
  local metrics="${DP_DIR}/data/kitchen_eval/${model}/seed_${seed}/eval_metrics.json"
  if [[ -f "${metrics}" ]]; then
    local p
    p="$(ckpt_from_metrics "${metrics}")"
    if [[ -n "${p}" && -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  fi
  # fallback: best test_mean_score under train dir
  python3 -c "
import re, pathlib
d=pathlib.Path(r'''${DP_DIR}/data/${model}/${seed}''')
best=None; best_s=float('-inf')
if d.is_dir():
  for p in d.glob('epoch=*.ckpt'):
    m=re.search(r'test_mean_score=(-?[0-9.]+)', p.name)
    if not m: continue
    s=float(m.group(1))
    if s>=best_s: best_s=s; best=str(p)
  if best: print(best)
  elif (d/'latest-001.ckpt').is_file(): print(d/'latest-001.ckpt')
" 2>/dev/null || true
}

resolve_fp_ckpt() {
  local seed="$1"  # baseline_42 | baseline_43 | baseline_44
  local metrics="${FP_DIR}/data/kitchen_eval/flowpolicy/seed_${seed}/eval_metrics.json"
  if [[ -f "${metrics}" ]]; then
    local p
    p="$(ckpt_from_metrics "${metrics}")"
    if [[ -n "${p}" && -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  fi
  local d="${FP_DIR}/data/outputs/${seed}"
  if [[ -f "${d}/latest-001.ckpt" ]]; then
    echo "${d}/latest-001.ckpt"
    return 0
  fi
  # baseline_44 sometimes has latest-001(1).ckpt
  ls -1 "${d}"/latest*.ckpt 2>/dev/null | sort | tail -1 || true
}

overwrite_flag() {
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    echo "--overwrite"
  fi
}

should_run_model() {
  local kind="$1"  # cnn|transformer|flowpolicy
  [[ "${MODEL_FILTER}" == "all" || "${MODEL_FILTER}" == "${kind}" ]]
}

nfe_list_effective() {
  if [[ -n "${NFE_FILTER}" ]]; then
    echo "${NFE_FILTER}"
  else
    printf '%s\n' "${NFE_LIST[@]}"
  fi
}

run_dp() {
  local model="$1"      # diffusion_policy_cnn | diffusion_policy_transformer
  local seed_name="$2"  # train0|1|2
  local ckpt="$3"
  local nfe="$4"
  local out_dir="${OUT_DP}/${model}/seed_${seed_name}_nfe${nfe}_sseed${SAMPLING_SEED}"

  if [[ -f "${out_dir}/eval_metrics.json" && "${OVERWRITE}" -eq 0 ]]; then
    log "SKIP ${model} ${seed_name} nfe=${nfe} (exists)"
    return 0
  fi
  if [[ -d "${out_dir}" && ! -f "${out_dir}/eval_metrics.json" ]]; then
    log "WARN: removing incomplete ${out_dir}"
    rm -rf "${out_dir}"
  fi

  log "RUN ${model} seed=${seed_name} nfe=${nfe} eps=${N_EPISODES}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "  DRY-RUN ckpt=${ckpt}"
    log "  DRY-RUN out=${out_dir}"
    return 0
  fi

  cd "${DP_DIR}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate robodiff
  export MUJOCO_GL=egl
  # shellcheck disable=SC2046
  python eval_kitchen.py \
    --model "${model}" \
    -c "${ckpt}" \
    --output_root "${OUT_DP}" \
    --n_episodes "${N_EPISODES}" \
    --device "${DEVICE}" \
    --num_inference_steps "${nfe}" \
    --sampling_seed "${SAMPLING_SEED}" \
    --video \
    --save-trajectory-logs \
    $(overwrite_flag) \
    2>&1 | tee -a "${LOG_FILE}"
}

run_fp() {
  local seed_name="$1"  # baseline_42|43|44
  local ckpt="$2"
  local nfe="$3"
  local out_dir="${OUT_FP}/seed_${seed_name}_nfe${nfe}_sseed${SAMPLING_SEED}"

  if [[ -f "${out_dir}/eval_metrics.json" && "${OVERWRITE}" -eq 0 ]]; then
    log "SKIP flowpolicy ${seed_name} nfe=${nfe} (exists)"
    return 0
  fi
  if [[ -d "${out_dir}" && ! -f "${out_dir}/eval_metrics.json" ]]; then
    log "WARN: removing incomplete ${out_dir}"
    rm -rf "${out_dir}"
  fi

  log "RUN flowpolicy seed=${seed_name} nfe=${nfe} eps=${N_EPISODES}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "  DRY-RUN ckpt=${ckpt}"
    log "  DRY-RUN out=${out_dir}"
    return 0
  fi

  cd "${FP_DIR}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate flowpolicy-kitchen
  export MUJOCO_GL=egl
  # shellcheck disable=SC2046
  python eval_kitchen.py \
    -c "${ckpt}" \
    --output_root "${OUT_FP}" \
    --n_episodes "${N_EPISODES}" \
    --device "${DEVICE}" \
    --num_inference_steps "${nfe}" \
    --sampling_seed "${SAMPLING_SEED}" \
    --video \
    --save-trajectory-logs \
    $(overwrite_flag) \
    2>&1 | tee -a "${LOG_FILE}"
}

# --- analyze only ---
if [[ "${ANALYZE_ONLY}" -eq 1 ]]; then
  cd "${EXPERIMENT_ROOT}"
  python3 scripts/analyze_kitchen_nfe100.py
  exit 0
fi

log "Kitchen NFE100 full eval start. log=${LOG_FILE}"
log "episodes=${N_EPISODES} device=${DEVICE} sampling_seed=${SAMPLING_SEED} dry_run=${DRY_RUN} model=${MODEL_FILTER}"

# Preflight datasets
if [[ ! -f "${DP_DIR}/data/kitchen/all_init_qpos.npy" ]]; then
  log "ERROR: missing DP Kitchen dataset"
  exit 1
fi
if [[ ! -f "${FP_DIR}/data/kitchen/all_init_qpos.npy" ]]; then
  log "ERROR: missing FlowPolicy Kitchen dataset"
  exit 1
fi

# Disk warning (best-effort)
avail_gb="$(df -BG --output=avail "${EXPERIMENT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
if [[ "${avail_gb}" =~ ^[0-9]+$ ]] && [[ "${avail_gb}" -lt 20 ]]; then
  log "WARN: only ~${avail_gb}G free; full run may need ~10G+ for videos/traj"
fi

# Resolve and validate checkpoints up front
DP_SEEDS=(train0 train1 train2)
FP_SEEDS=(baseline_42 baseline_43 baseline_44)

declare -A DP_CNN_CKPT DP_TF_CKPT FP_CKPT

if should_run_model cnn; then
  for s in "${DP_SEEDS[@]}"; do
    DP_CNN_CKPT[$s]="$(resolve_dp_ckpt diffusion_policy_cnn "${s}")"
    if [[ -z "${DP_CNN_CKPT[$s]}" || ! -f "${DP_CNN_CKPT[$s]}" ]]; then
      log "ERROR: missing DP-CNN ckpt for ${s}"
      exit 1
    fi
    log "DP-CNN ${s}: ${DP_CNN_CKPT[$s]}"
  done
fi

if should_run_model transformer; then
  for s in "${DP_SEEDS[@]}"; do
    DP_TF_CKPT[$s]="$(resolve_dp_ckpt diffusion_policy_transformer "${s}")"
    if [[ -z "${DP_TF_CKPT[$s]}" || ! -f "${DP_TF_CKPT[$s]}" ]]; then
      log "ERROR: missing DP-Transformer ckpt for ${s}"
      exit 1
    fi
    log "DP-Transformer ${s}: ${DP_TF_CKPT[$s]}"
  done
fi

if should_run_model flowpolicy; then
  for s in "${FP_SEEDS[@]}"; do
    FP_CKPT[$s]="$(resolve_fp_ckpt "${s}")"
    if [[ -z "${FP_CKPT[$s]}" || ! -f "${FP_CKPT[$s]}" ]]; then
      log "ERROR: missing FlowPolicy ckpt for ${s}"
      exit 1
    fi
    log "FlowPolicy ${s}: ${FP_CKPT[$s]}"
  done
fi

mapfile -t NFES < <(nfe_list_effective)

# --- runs ---
if should_run_model cnn; then
  for s in "${DP_SEEDS[@]}"; do
    for nfe in "${NFES[@]}"; do
      run_dp diffusion_policy_cnn "${s}" "${DP_CNN_CKPT[$s]}" "${nfe}"
    done
  done
fi

if should_run_model transformer; then
  for s in "${DP_SEEDS[@]}"; do
    for nfe in "${NFES[@]}"; do
      run_dp diffusion_policy_transformer "${s}" "${DP_TF_CKPT[$s]}" "${nfe}"
    done
  done
fi

if should_run_model flowpolicy; then
  for s in "${FP_SEEDS[@]}"; do
    for nfe in "${NFES[@]}"; do
      run_fp "${s}" "${FP_CKPT[$s]}" "${nfe}"
    done
  done
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "DRY-RUN complete (no eval / no analyze)."
  exit 0
fi

log "All scheduled runs finished. Analyzing..."
cd "${EXPERIMENT_ROOT}"
python3 scripts/analyze_kitchen_nfe100.py 2>&1 | tee -a "${LOG_FILE}"
log "Done. Report: ${EXPERIMENT_ROOT}/data/kitchen_eval_plots/nfe100/report.txt"
