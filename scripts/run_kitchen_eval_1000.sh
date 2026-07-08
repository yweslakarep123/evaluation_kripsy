#!/usr/bin/env bash
# Run flat Kitchen evaluation: 100 episodes per checkpoint.
#
# Prerequisites:
#   - conda env robodiff (Diffusion Policy) with working Kitchen/MuJoCo
#   - conda env flowpolicy-kitchen (FlowPolicy)
#
# Usage:
#   bash scripts/run_kitchen_eval_1000.sh
#   bash scripts/run_kitchen_eval_1000.sh --smoke
#   bash scripts/run_kitchen_eval_1000.sh --overwrite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/kitchen_eval_100_${TIMESTAMP}.log"

SMOKE=""
OVERWRITE=""
DEVICE="${DEVICE:-cuda:0}"
N_EPISODES=100
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE="--smoke" ;;
        --overwrite) OVERWRITE="--overwrite" ;;
        --device) DEVICE="$2"; shift ;;
    esac
    shift
done

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

has_dp_checkpoints() {
    local model="$1"
    local ckpt_dir="${EXPERIMENT_ROOT}/diffusion_policy/data/${model}/train0"
    compgen -G "${ckpt_dir}/epoch=*.ckpt" > /dev/null 2>&1 \
        || compgen -G "${ckpt_dir}/*.ckpt" > /dev/null 2>&1
}

run_dp() {
    local model="$1"
    if ! has_dp_checkpoints "${model}"; then
        log "Skipping ${model}: no checkpoints in diffusion_policy/data/${model}/train{0,1,2}/"
        return 0
    fi
    log "Starting Diffusion Policy: ${model}"
    cd "${EXPERIMENT_ROOT}/diffusion_policy"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate robodiff
    export MUJOCO_GL=egl
    python eval_kitchen.py \
        --model "${model}" \
        --output_root data/kitchen_eval \
        --n_episodes "${N_EPISODES}" \
        --device "${DEVICE}" \
        ${SMOKE} \
        ${OVERWRITE} \
        2>&1 | tee -a "${LOG_FILE}"
}

run_flowpolicy() {
    log "Starting FlowPolicy"
    cd "${EXPERIMENT_ROOT}/kripsy12/FlowPolicy"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate flowpolicy-kitchen
    export MUJOCO_GL=egl
    python eval_kitchen.py \
        --output_root data/kitchen_eval/flowpolicy \
        --n_episodes "${N_EPISODES}" \
        --device "${DEVICE}" \
        ${SMOKE} \
        ${OVERWRITE} \
        2>&1 | tee -a "${LOG_FILE}"
}

log "Kitchen flat eval started. Log: ${LOG_FILE}"
log "Episodes per checkpoint: ${N_EPISODES}"
if [[ -n "${SMOKE}" ]]; then
    log "SMOKE TEST mode (10 episodes, 1 checkpoint per model)"
fi

run_dp diffusion_policy_transformer
run_dp diffusion_policy_cnn
run_flowpolicy

log "All evaluations complete."
