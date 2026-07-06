#!/usr/bin/env bash
# Run full Kitchen combinatorial evaluation for all models.
#
# Prerequisites:
#   - conda env robodiff (Diffusion Policy) with working Kitchen/MuJoCo
#   - conda env flowpolicy-kitchen (FlowPolicy) — Kitchen env may need
#     dm_control/mujoco pin; see kripsy12/ReinFlow/docs/KnownIssues.md
#
# Usage:
#   bash scripts/run_all_kitchen_combo_eval.sh
#   bash scripts/run_all_kitchen_combo_eval.sh --smoke

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/kitchen_combo_eval_${TIMESTAMP}.log"

SMOKE=""
DEVICE="${DEVICE:-cuda:0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE="--smoke" ;;
        --device) DEVICE="$2"; shift ;;
    esac
    shift
done

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

run_dp() {
    local model="$1"
    log "Starting Diffusion Policy: ${model}"
    cd "${EXPERIMENT_ROOT}/diffusion_policy"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate robodiff
    export MUJOCO_GL=egl
    python eval_kitchen_combinations.py \
        --model "${model}" \
        --output_root data/kitchen_combo_eval \
        --device "${DEVICE}" \
        --resume \
        ${SMOKE} \
        2>&1 | tee -a "${LOG_FILE}"
}

run_flowpolicy() {
    log "Starting FlowPolicy"
    cd "${EXPERIMENT_ROOT}/kripsy12/FlowPolicy"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate flowpolicy-kitchen
    export MUJOCO_GL=egl
    python eval_kitchen_combinations.py \
        --output_root data/kitchen_combo_eval/flowpolicy \
        --device "${DEVICE}" \
        --resume \
        ${SMOKE} \
        2>&1 | tee -a "${LOG_FILE}"
}

log "Kitchen combinatorial eval started. Log: ${LOG_FILE}"
if [[ -n "${SMOKE}" ]]; then
    log "SMOKE TEST mode"
fi

run_dp diffusion_policy_transformer
run_dp diffusion_policy_cnn
run_flowpolicy

log "All evaluations complete."
