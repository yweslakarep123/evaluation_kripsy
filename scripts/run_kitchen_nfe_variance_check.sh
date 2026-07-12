#!/usr/bin/env bash
# Lean Kitchen sweep: success vs NFE + sampling-seed variance.
#
# Default: 1 checkpoint/model, 50 episodes, no video, no trajectory logs.
# Does NOT touch the main kitchen_eval/ results.
#
# Usage (from experiment root):
#   bash scripts/run_kitchen_nfe_variance_check.sh
#   bash scripts/run_kitchen_nfe_variance_check.sh --skip-transformer
#   bash scripts/run_kitchen_nfe_variance_check.sh --nfe-only
#   bash scripts/run_kitchen_nfe_variance_check.sh --variance-only
#   bash scripts/run_kitchen_nfe_variance_check.sh --dry-run
#   N_EPISODES=30 DEVICE=cuda:0 bash scripts/run_kitchen_nfe_variance_check.sh
#
# After the sweep finishes, analyze-only:
#   python scripts/analyze_kitchen_nfe_variance.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/kitchen_nfe_variance_${TIMESTAMP}.log"

DEVICE="${DEVICE:-cuda:0}"
N_EPISODES="${N_EPISODES:-50}"
OUTPUT_ROOT_DP="${EXPERIMENT_ROOT}/diffusion_policy/data/kitchen_eval_nfe"
OUTPUT_ROOT_FP="${EXPERIMENT_ROOT}/kripsy12/FlowPolicy/data/kitchen_eval_nfe/flowpolicy"

NFE_ONLY=0
VARIANCE_ONLY=0
SKIP_TRANSFORMER=0
DRY_RUN=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nfe-only) NFE_ONLY=1 ;;
        --variance-only) VARIANCE_ONLY=1 ;;
        --skip-transformer) SKIP_TRANSFORMER=1 ;;
        --overwrite) OVERWRITE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --device)
            DEVICE="$2"
            shift
            ;;
        --n_episodes)
            N_EPISODES="$2"
            shift
            ;;
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

if [[ "${NFE_ONLY}" -eq 1 && "${VARIANCE_ONLY}" -eq 1 ]]; then
    echo "ERROR: use only one of --nfe-only / --variance-only" >&2
    exit 1
fi

# NFE grids (plan)
DP_NFE_LIST=(1 2 4 8 16 32 50 100)
FP_NFE_LIST=(1 2 4)
SSEED_LIST=(0 1 2 3 4)
DEFAULT_DP_NFE=100
DEFAULT_FP_NFE=1

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT_DP}" "${OUTPUT_ROOT_FP}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Prefer the exact checkpoint used by the main Kitchen eval (eval_metrics.json).
# Fallback: highest test_mean_score in filename, then latest-001.ckpt.
resolve_dp_ckpt() {
    local model="$1"
    local metrics="${EXPERIMENT_ROOT}/diffusion_policy/data/kitchen_eval/${model}/seed_train0/eval_metrics.json"
    if [[ -f "${metrics}" ]]; then
        local from_metrics
        from_metrics="$(python3 -c "
import json,sys
p=json.load(open(sys.argv[1])).get('checkpoint','')
print(p if p else '')
" "${metrics}" 2>/dev/null || true)"
        if [[ -n "${from_metrics}" && -f "${from_metrics}" ]]; then
            echo "${from_metrics}"
            return 0
        fi
    fi

    local train_dir="${EXPERIMENT_ROOT}/diffusion_policy/data/${model}/train0"
    if [[ ! -d "${train_dir}" ]]; then
        echo ""
        return 0
    fi

    local best
    best="$(python3 -c "
import re, pathlib
d=pathlib.Path(r'''${train_dir}''')
best=None
best_score=float('-inf')
for p in d.glob('epoch=*.ckpt'):
    m=re.search(r'test_mean_score=(-?[0-9.]+)', p.name)
    if not m:
        continue
    s=float(m.group(1))
    if s>=best_score:
        best_score=s
        best=str(p)
if best:
    print(best)
elif (d/'latest-001.ckpt').is_file():
    print(d/'latest-001.ckpt')
else:
    ckpts=sorted(d.glob('*.ckpt'))
    print(ckpts[-1] if ckpts else '')
" 2>/dev/null || true)"
    echo "${best}"
}

resolve_fp_ckpt() {
    local metrics="${EXPERIMENT_ROOT}/kripsy12/FlowPolicy/data/kitchen_eval/flowpolicy/seed_baseline_42/eval_metrics.json"
    if [[ -f "${metrics}" ]]; then
        local from_metrics
        from_metrics="$(python3 -c "
import json,sys
p=json.load(open(sys.argv[1])).get('checkpoint','')
print(p if p else '')
" "${metrics}" 2>/dev/null || true)"
        if [[ -n "${from_metrics}" && -f "${from_metrics}" ]]; then
            echo "${from_metrics}"
            return 0
        fi
    fi
    local ckpt="${EXPERIMENT_ROOT}/kripsy12/FlowPolicy/data/outputs/baseline_42/latest-001.ckpt"
    if [[ -f "${ckpt}" ]]; then
        echo "${ckpt}"
        return 0
    fi
    ls -1 "${EXPERIMENT_ROOT}/kripsy12/FlowPolicy/data/outputs/baseline_42/"*.ckpt 2>/dev/null | sort | tail -1 || true
}

overwrite_flag() {
    if [[ "${OVERWRITE}" -eq 1 ]]; then
        echo "--overwrite"
    fi
}

run_dp_one() {
    local model="$1"
    local ckpt="$2"
    local nfe="$3"
    local sseed="$4"
    local seed_name
    seed_name="$(basename "$(dirname "${ckpt}")")"
    local out_dir="${OUTPUT_ROOT_DP}/${model}/seed_${seed_name}_nfe${nfe}_sseed${sseed}"
    if [[ -f "${out_dir}/eval_metrics.json" && "${OVERWRITE}" -eq 0 ]]; then
        log "SKIP DP ${model} nfe=${nfe} sseed=${sseed} (exists)"
        return 0
    fi
    log "DP ${model} nfe=${nfe} sseed=${sseed} eps=${N_EPISODES}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "  DRY-RUN would eval: ${ckpt} -> ${out_dir}"
        return 0
    fi
    cd "${EXPERIMENT_ROOT}/diffusion_policy"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate robodiff
    export MUJOCO_GL=egl
    # shellcheck disable=SC2046
    python eval_kitchen.py \
        --model "${model}" \
        -c "${ckpt}" \
        --output_root "${OUTPUT_ROOT_DP}" \
        --n_episodes "${N_EPISODES}" \
        --device "${DEVICE}" \
        --num_inference_steps "${nfe}" \
        --sampling_seed "${sseed}" \
        --no-video \
        --no-save-trajectory-logs \
        $(overwrite_flag) \
        2>&1 | tee -a "${LOG_FILE}"
}

run_fp_one() {
    local ckpt="$1"
    local nfe="$2"
    local sseed="$3"
    local seed_name
    seed_name="$(basename "$(dirname "${ckpt}")")"
    local out_dir="${OUTPUT_ROOT_FP}/seed_${seed_name}_nfe${nfe}_sseed${sseed}"
    if [[ -f "${out_dir}/eval_metrics.json" && "${OVERWRITE}" -eq 0 ]]; then
        log "SKIP FlowPolicy nfe=${nfe} sseed=${sseed} (exists)"
        return 0
    fi
    log "FlowPolicy nfe=${nfe} sseed=${sseed} eps=${N_EPISODES}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "  DRY-RUN would eval: ${ckpt} -> ${out_dir}"
        return 0
    fi
    cd "${EXPERIMENT_ROOT}/kripsy12/FlowPolicy"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate flowpolicy-kitchen
    export MUJOCO_GL=egl
    # shellcheck disable=SC2046
    python eval_kitchen.py \
        -c "${ckpt}" \
        --output_root "${OUTPUT_ROOT_FP}" \
        --n_episodes "${N_EPISODES}" \
        --device "${DEVICE}" \
        --num_inference_steps "${nfe}" \
        --sampling_seed "${sseed}" \
        --no-video \
        --no-save-trajectory-logs \
        $(overwrite_flag) \
        2>&1 | tee -a "${LOG_FILE}"
}

log "Kitchen NFE/variance check started. Log: ${LOG_FILE}"
log "Episodes: ${N_EPISODES}  Device: ${DEVICE}  dry_run=${DRY_RUN}"

# Preflight: datasets
DP_DS="${EXPERIMENT_ROOT}/diffusion_policy/data/kitchen/all_init_qpos.npy"
FP_DS="${EXPERIMENT_ROOT}/kripsy12/FlowPolicy/data/kitchen/all_init_qpos.npy"
if [[ ! -f "${DP_DS}" ]]; then
    log "ERROR: missing DP dataset: ${DP_DS}"
    exit 1
fi
if [[ ! -f "${FP_DS}" ]]; then
    log "ERROR: missing FlowPolicy dataset: ${FP_DS}"
    exit 1
fi

DP_CNN_CKPT="$(resolve_dp_ckpt diffusion_policy_cnn)"
DP_TF_CKPT="$(resolve_dp_ckpt diffusion_policy_transformer)"
FP_CKPT="$(resolve_fp_ckpt)"

if [[ -z "${DP_CNN_CKPT}" || ! -f "${DP_CNN_CKPT}" ]]; then
    log "ERROR: no DP-CNN checkpoint for train0"
    exit 1
fi
if [[ -z "${FP_CKPT}" || ! -f "${FP_CKPT}" ]]; then
    log "ERROR: no FlowPolicy checkpoint for baseline_42"
    exit 1
fi

log "DP-CNN ckpt: ${DP_CNN_CKPT}"
log "FlowPolicy ckpt: ${FP_CKPT}"
if [[ "${SKIP_TRANSFORMER}" -eq 0 ]]; then
    if [[ -z "${DP_TF_CKPT}" || ! -f "${DP_TF_CKPT}" ]]; then
        log "WARNING: no DP-Transformer ckpt; skipping Transformer"
        SKIP_TRANSFORMER=1
    else
        log "DP-Transformer ckpt: ${DP_TF_CKPT}"
    fi
fi

# --- Claim #1: NFE sweep (fixed sampling_seed=0) ---
if [[ "${VARIANCE_ONLY}" -eq 0 ]]; then
    log "=== Claim #1: NFE sweep (sseed=0) ==="
    for nfe in "${DP_NFE_LIST[@]}"; do
        run_dp_one diffusion_policy_cnn "${DP_CNN_CKPT}" "${nfe}" 0
        if [[ "${SKIP_TRANSFORMER}" -eq 0 ]]; then
            run_dp_one diffusion_policy_transformer "${DP_TF_CKPT}" "${nfe}" 0
        fi
    done
    for nfe in "${FP_NFE_LIST[@]}"; do
        run_fp_one "${FP_CKPT}" "${nfe}" 0
    done
fi

# --- Claim #2: sampling-seed variance at default NFE ---
if [[ "${NFE_ONLY}" -eq 0 ]]; then
    log "=== Claim #2: sampling-seed variance at default NFE ==="
    for sseed in "${SSEED_LIST[@]}"; do
        # sseed=0 @ default NFE already produced by NFE sweep
        if [[ "${VARIANCE_ONLY}" -eq 0 && "${sseed}" -eq 0 ]]; then
            log "Skipping sseed=0 (already from NFE sweep)"
            continue
        fi
        run_dp_one diffusion_policy_cnn "${DP_CNN_CKPT}" "${DEFAULT_DP_NFE}" "${sseed}"
        if [[ "${SKIP_TRANSFORMER}" -eq 0 ]]; then
            run_dp_one diffusion_policy_transformer "${DP_TF_CKPT}" "${DEFAULT_DP_NFE}" "${sseed}"
        fi
        run_fp_one "${FP_CKPT}" "${DEFAULT_FP_NFE}" "${sseed}"
    done
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN complete (no eval / no analyze)."
    exit 0
fi

log "Sweep complete. Analyzing..."
cd "${EXPERIMENT_ROOT}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robodiff
python scripts/analyze_kitchen_nfe_variance.py \
    --input_root_dp "${OUTPUT_ROOT_DP}" \
    --input_root_fp "${OUTPUT_ROOT_FP}" \
    --output_dir "${EXPERIMENT_ROOT}/data/kitchen_eval_plots/nfe_variance" \
    2>&1 | tee -a "${LOG_FILE}"

log "Done. Report: ${EXPERIMENT_ROOT}/data/kitchen_eval_plots/nfe_variance/report.txt"
