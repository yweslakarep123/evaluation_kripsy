#!/usr/bin/env bash
# Full Kitchen eval (best_val FlowPolicy + DP@100 + native LSTM/IBC/BeT).
#
# Models:
#   DP-CNN / DP-Transformer: train0, train1, train2  — NFE=100 only
#   FlowPolicy:              seed42/43/44 best_val.ckpt — NFE 1..32 and 100
#   LSTM-GMM / IBC / BeT:    train0/1/2 — default config (no NFE override)
# Fairness note: DP is not evaluated below NFE=100.
#
# Output: data/kitchen_eval_nfe100/<model>/...
# Does NOT auto-analyze at the end (use --analyze-only after the sweep).
#
# Usage:
#   bash scripts/run_kitchen_nfe100_full.sh --dry-run
#   bash scripts/run_kitchen_nfe100_full.sh
#   bash scripts/run_kitchen_nfe100_full.sh --model flowpolicy
#   bash scripts/run_kitchen_nfe100_full.sh --model flowpolicy --nfe 1
#   bash scripts/run_kitchen_nfe100_full.sh --model flowpolicy --nfe-list 1-8
#   bash scripts/run_kitchen_nfe100_full.sh --analyze-only
#   VIDEO=1 DEVICE=cuda:0 bash scripts/run_kitchen_nfe100_full.sh --overwrite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DP_DIR="${EXPERIMENT_ROOT}/diffusion_policy"
FP_DIR="${EXPERIMENT_ROOT}/kripsy12/FlowPolicy"
OUT_ROOT="${EXPERIMENT_ROOT}/data/kitchen_eval_nfe100"
OUT_DP="${OUT_ROOT}"
OUT_FP="${OUT_ROOT}/flowpolicy"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/kitchen_nfe100_$(date +%Y%m%d_%H%M%S).log"
PROGRESS_FILE="${OUT_ROOT}/progress.jsonl"

N_EPISODES="${N_EPISODES:-100}"
DEVICE="${DEVICE:-cuda:0}"
SAMPLING_SEED=0
WARMUP_CALLS="${WARMUP_CALLS:-10}"
VIDEO="${VIDEO:-0}"
# FlowPolicy keeps the full trade-off grid; Diffusion Policy only NFE=100.
FP_NFE_LIST=($(seq 1 32) 100)
DP_NFE_LIST=(100)
MODEL_FILTER="all"
NFE_FILTER=""
NFE_LIST_RAW=""
DRY_RUN=0
ANALYZE_ONLY=0
OVERWRITE=0

usage() {
  sed -n '2,24p' "$0"
}

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
        cnn|transformer|flowpolicy|lstm|ibc|bet|all) ;;
        *)
          echo "Unknown --model ${MODEL_FILTER} (cnn|transformer|flowpolicy|lstm|ibc|bet|all)" >&2
          exit 1
          ;;
      esac
      ;;
    --nfe) NFE_FILTER="$2"; shift ;;
    --nfe-list) NFE_LIST_RAW="$2"; shift ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1 (try --help)" >&2
      exit 1
      ;;
  esac
  shift
done

if [[ -n "${NFE_LIST_RAW}" ]]; then
  if [[ "${NFE_LIST_RAW}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    FP_NFE_LIST=($(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"))
  else
    IFS=', ' read -r -a FP_NFE_LIST <<< "${NFE_LIST_RAW}"
  fi
fi

mkdir -p "${LOG_DIR}" "${OUT_ROOT}" "${OUT_FP}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

progress() {
  local status="$1" model="$2" seed="$3" nfe="$4" out_dir="$5"
  python3 -c "
import json, datetime, sys
print(json.dumps({
  'ts': datetime.datetime.now().isoformat(timespec='seconds'),
  'status': sys.argv[1],
  'model': sys.argv[2],
  'seed': sys.argv[3],
  'nfe': sys.argv[4],
  'out_dir': sys.argv[5],
}))
" "${status}" "${model}" "${seed}" "${nfe}" "${out_dir}" >> "${PROGRESS_FILE}"
}

metrics_complete() {
  local metrics="$1"
  python3 -c "
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
try:
    m = json.load(open(path))
except Exception:
    raise SystemExit(1)
got = m.get('n_episodes')
if got is None:
    got = len(m.get('episodes') or [])
try:
    raise SystemExit(0 if int(got) == expected else 1)
except (TypeError, ValueError):
    raise SystemExit(1)
" "${metrics}" "${N_EPISODES}"
}

resolve_dp_ckpt() {
  local model="$1"
  local seed="$2"
  python3 -c "
import re, pathlib
d = pathlib.Path(r'''${DP_DIR}/data/${model}/${seed}''')
best = None
best_s = float('-inf')
if d.is_dir():
    for p in d.glob('epoch=*.ckpt'):
        m = re.search(r'test_mean_score=(-?[0-9]+(?:\\.[0-9]+)?)', p.name)
        if not m:
            continue
        s = float(m.group(1))
        if s >= best_s:
            best_s = s
            best = str(p)
    if best:
        print(best)
    elif (d / 'latest-001.ckpt').is_file():
        print(d / 'latest-001.ckpt')
" 2>/dev/null || true
}

video_flag() {
  if [[ "${VIDEO}" == "1" ]]; then
    echo "--video"
  else
    echo "--no-video"
  fi
}

overwrite_flag() {
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    echo "--overwrite"
  fi
}

should_run_model() {
  local kind="$1"
  [[ "${MODEL_FILTER}" == "all" || "${MODEL_FILTER}" == "${kind}" ]]
}

nfe_list_effective() {
  local family="${1:-fp}"
  if [[ -n "${NFE_FILTER}" ]]; then
    if [[ "${family}" == "dp" && "${NFE_FILTER}" -ne 100 ]]; then
      echo "WARN: DP NFE<100 is disabled for fairness; ignoring --nfe ${NFE_FILTER}" >&2
      return 0
    fi
    echo "${NFE_FILTER}"
    return 0
  fi
  if [[ "${family}" == "dp" ]]; then
    printf '%s\n' "${DP_NFE_LIST[@]}"
  else
    printf '%s\n' "${FP_NFE_LIST[@]}"
  fi
}

activate_conda() {
  local env="$1"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${env}"
  export MUJOCO_GL=egl
}

maybe_skip_or_clean() {
  local out_dir="$1"
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    return 1
  fi
  if [[ -f "${out_dir}/eval_metrics.json" ]] && metrics_complete "${out_dir}/eval_metrics.json"; then
    return 0
  fi
  if [[ -d "${out_dir}" ]]; then
    log "WARN: removing incomplete ${out_dir}"
    rm -rf "${out_dir}"
  fi
  return 1
}

run_dp() {
  local model="$1"
  local seed_name="$2"
  local ckpt="$3"
  local nfe="$4"
  local out_dir="${OUT_DP}/${model}/seed_${seed_name}_nfe${nfe}_sseed${SAMPLING_SEED}"

  if maybe_skip_or_clean "${out_dir}"; then
    log "SKIP ${model} ${seed_name} nfe=${nfe} (exists)"
    progress skip "${model}" "${seed_name}" "${nfe}" "${out_dir}"
    return 0
  fi

  log "RUN ${model} seed=${seed_name} nfe=${nfe} eps=${N_EPISODES}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "  DRY-RUN ckpt=${ckpt}"
    log "  DRY-RUN out=${out_dir}"
    progress dry-run "${model}" "${seed_name}" "${nfe}" "${out_dir}"
    return 0
  fi

  cd "${DP_DIR}"
  activate_conda robodiff
  set +e
  # shellcheck disable=SC2046
  python eval_kitchen.py \
    --model "${model}" \
    -c "${ckpt}" \
    --output_root "${OUT_DP}" \
    --n_episodes "${N_EPISODES}" \
    --device "${DEVICE}" \
    --num_inference_steps "${nfe}" \
    --sampling_seed "${SAMPLING_SEED}" \
    --inference_warmup_calls "${WARMUP_CALLS}" \
    $(video_flag) \
    --save-trajectory-logs \
    $(overwrite_flag) \
    2>&1 | tee -a "${LOG_FILE}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 0 ]] && metrics_complete "${out_dir}/eval_metrics.json"; then
    progress ok "${model}" "${seed_name}" "${nfe}" "${out_dir}"
  else
    log "FAIL ${model} ${seed_name} nfe=${nfe} rc=${rc}"
    progress fail "${model}" "${seed_name}" "${nfe}" "${out_dir}"
    return 0
  fi
}

run_native() {
  local model="$1"
  local seed_name="$2"
  local ckpt="$3"
  local out_dir="${OUT_DP}/${model}/seed_${seed_name}_sseed${SAMPLING_SEED}"

  if maybe_skip_or_clean "${out_dir}"; then
    log "SKIP ${model} ${seed_name} default (exists)"
    progress skip "${model}" "${seed_name}" "default" "${out_dir}"
    return 0
  fi

  log "RUN ${model} seed=${seed_name} default-config eps=${N_EPISODES}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "  DRY-RUN ckpt=${ckpt}"
    log "  DRY-RUN out=${out_dir}"
    progress dry-run "${model}" "${seed_name}" "default" "${out_dir}"
    return 0
  fi

  cd "${DP_DIR}"
  activate_conda robodiff
  set +e
  # shellcheck disable=SC2046
  python eval_kitchen.py \
    --model "${model}" \
    -c "${ckpt}" \
    --output_root "${OUT_DP}" \
    --n_episodes "${N_EPISODES}" \
    --device "${DEVICE}" \
    --sampling_seed "${SAMPLING_SEED}" \
    --inference_warmup_calls "${WARMUP_CALLS}" \
    $(video_flag) \
    --save-trajectory-logs \
    $(overwrite_flag) \
    2>&1 | tee -a "${LOG_FILE}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 0 ]] && metrics_complete "${out_dir}/eval_metrics.json"; then
    progress ok "${model}" "${seed_name}" "default" "${out_dir}"
  else
    log "FAIL ${model} ${seed_name} default rc=${rc}"
    progress fail "${model}" "${seed_name}" "default" "${out_dir}"
    return 0
  fi
}

run_fp() {
  local seed_name="$1"
  local ckpt="$2"
  local nfe="$3"
  local out_dir="${OUT_FP}/seed_${seed_name}_nfe${nfe}_sseed${SAMPLING_SEED}"

  if maybe_skip_or_clean "${out_dir}"; then
    log "SKIP flowpolicy ${seed_name} nfe=${nfe} (exists)"
    progress skip flowpolicy "${seed_name}" "${nfe}" "${out_dir}"
    return 0
  fi

  log "RUN flowpolicy seed=${seed_name} nfe=${nfe} eps=${N_EPISODES}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "  DRY-RUN ckpt=${ckpt}"
    log "  DRY-RUN out=${out_dir}"
    progress dry-run flowpolicy "${seed_name}" "${nfe}" "${out_dir}"
    return 0
  fi

  cd "${FP_DIR}"
  activate_conda flowpolicy-kitchen
  set +e
  # shellcheck disable=SC2046
  python eval_kitchen.py \
    -c "${ckpt}" \
    --run_name "${seed_name}" \
    --output_root "${OUT_FP}" \
    --n_episodes "${N_EPISODES}" \
    --device "${DEVICE}" \
    --num_inference_steps "${nfe}" \
    --sampling_seed "${SAMPLING_SEED}" \
    --inference_warmup_calls "${WARMUP_CALLS}" \
    $(video_flag) \
    --save-trajectory-logs \
    $(overwrite_flag) \
    2>&1 | tee -a "${LOG_FILE}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 0 ]] && metrics_complete "${out_dir}/eval_metrics.json"; then
    progress ok flowpolicy "${seed_name}" "${nfe}" "${out_dir}"
  else
    log "FAIL flowpolicy ${seed_name} nfe=${nfe} rc=${rc}"
    progress fail flowpolicy "${seed_name}" "${nfe}" "${out_dir}"
    return 0
  fi
}

# --- analyze only ---
if [[ "${ANALYZE_ONLY}" -eq 1 ]]; then
  cd "${EXPERIMENT_ROOT}"
  python3 scripts/analyze_kitchen_nfe100.py \
    --input_root_dp "${OUT_DP}" \
    --input_root_fp "${OUT_FP}" \
    --output_dir "${EXPERIMENT_ROOT}/data/kitchen_eval_plots/nfe100"
  exit 0
fi

log "Kitchen NFE100 best_val eval start. log=${LOG_FILE}"
log "episodes=${N_EPISODES} device=${DEVICE} sampling_seed=${SAMPLING_SEED} warmup=${WARMUP_CALLS} video=${VIDEO} dry_run=${DRY_RUN} model=${MODEL_FILTER}"
log "output=${OUT_ROOT}"

if [[ ! -f "${DP_DIR}/data/kitchen/all_init_qpos.npy" ]]; then
  log "ERROR: missing DP Kitchen dataset"
  exit 1
fi
if [[ ! -f "${FP_DIR}/data/kitchen/all_init_qpos.npy" ]]; then
  log "ERROR: missing FlowPolicy Kitchen dataset"
  exit 1
fi

avail_gb="$(df -BG --output=avail "${EXPERIMENT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
if [[ "${avail_gb}" =~ ^[0-9]+$ ]] && [[ "${avail_gb}" -lt 20 ]]; then
  log "WARN: only ~${avail_gb}G free; trajectory NPZ may need several GB"
fi

DP_SEEDS=(train0 train1 train2)
FP_SEEDS=(seed42 seed43 seed44)
declare -A FP_CKPT_PATH=(
  [seed42]="${FP_DIR}/data/outputs/seed42/runs/baseline_seed42_standard/checkpoints/best_val.ckpt"
  [seed43]="${FP_DIR}/data/outputs/seed43/runs/baseline_seed43_standard/checkpoints/best_val.ckpt"
  [seed44]="${FP_DIR}/data/outputs/seed44/runs/checkpoints/best_val.ckpt"
)

declare -A DP_CNN_CKPT DP_TF_CKPT LSTM_CKPT IBC_CKPT BET_CKPT FP_CKPT

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

if should_run_model lstm; then
  for s in "${DP_SEEDS[@]}"; do
    LSTM_CKPT[$s]="$(resolve_dp_ckpt LSTM_GMM "${s}")"
    if [[ -z "${LSTM_CKPT[$s]}" || ! -f "${LSTM_CKPT[$s]}" ]]; then
      log "ERROR: missing LSTM-GMM ckpt for ${s}"
      exit 1
    fi
    log "LSTM-GMM ${s}: ${LSTM_CKPT[$s]}"
  done
fi

if should_run_model ibc; then
  for s in "${DP_SEEDS[@]}"; do
    IBC_CKPT[$s]="$(resolve_dp_ckpt implicit_behavior_cloning "${s}")"
    if [[ -z "${IBC_CKPT[$s]}" || ! -f "${IBC_CKPT[$s]}" ]]; then
      log "ERROR: missing IBC ckpt for ${s}"
      exit 1
    fi
    log "IBC ${s}: ${IBC_CKPT[$s]}"
  done
fi

if should_run_model bet; then
  for s in "${DP_SEEDS[@]}"; do
    BET_CKPT[$s]="$(resolve_dp_ckpt behavior_transformer "${s}")"
    if [[ -z "${BET_CKPT[$s]}" || ! -f "${BET_CKPT[$s]}" ]]; then
      log "ERROR: missing BeT ckpt for ${s}"
      exit 1
    fi
    log "BeT ${s}: ${BET_CKPT[$s]}"
  done
fi

if should_run_model flowpolicy; then
  for s in "${FP_SEEDS[@]}"; do
    FP_CKPT[$s]="${FP_CKPT_PATH[$s]}"
    if [[ ! -f "${FP_CKPT[$s]}" ]]; then
      log "ERROR: missing FlowPolicy best_val.ckpt for ${s}: ${FP_CKPT[$s]}"
      exit 1
    fi
    log "FlowPolicy ${s}: ${FP_CKPT[$s]}"
  done
fi

mapfile -t DP_NFES < <(nfe_list_effective dp)
mapfile -t FP_NFES < <(nfe_list_effective fp)

if should_run_model cnn; then
  for s in "${DP_SEEDS[@]}"; do
    for nfe in "${DP_NFES[@]}"; do
      run_dp diffusion_policy_cnn "${s}" "${DP_CNN_CKPT[$s]}" "${nfe}"
    done
  done
fi

if should_run_model transformer; then
  for s in "${DP_SEEDS[@]}"; do
    for nfe in "${DP_NFES[@]}"; do
      run_dp diffusion_policy_transformer "${s}" "${DP_TF_CKPT[$s]}" "${nfe}"
    done
  done
fi

if should_run_model lstm; then
  for s in "${DP_SEEDS[@]}"; do
    run_native LSTM_GMM "${s}" "${LSTM_CKPT[$s]}"
  done
fi

if should_run_model ibc; then
  for s in "${DP_SEEDS[@]}"; do
    run_native implicit_behavior_cloning "${s}" "${IBC_CKPT[$s]}"
  done
fi

if should_run_model bet; then
  for s in "${DP_SEEDS[@]}"; do
    run_native behavior_transformer "${s}" "${BET_CKPT[$s]}"
  done
fi

if should_run_model flowpolicy; then
  for s in "${FP_SEEDS[@]}"; do
    for nfe in "${FP_NFES[@]}"; do
      run_fp "${s}" "${FP_CKPT[$s]}" "${nfe}"
    done
  done
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "DRY-RUN complete (no eval / no analyze)."
  exit 0
fi

log "All scheduled runs finished. NOT analyzing automatically."
log "When the sweep is complete, run:"
log "  bash scripts/run_kitchen_nfe100_full.sh --analyze-only"
log "Progress: ${PROGRESS_FILE}"
