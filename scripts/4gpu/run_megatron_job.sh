#!/usr/bin/env bash
# Generic Megatron launcher for 4-GPU experiment.
# Usage: run_megatron_job.sh <run_id> <nproc> <tp> <pp> [extra env already set]
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export EXPERIMENT=4gpu
source "${LAB_ROOT}/scripts/common.sh"

RUN_ID="${1:?run_id}"
NPROC="${2:?nproc}"
TP="${3:?tp}"
PP="${4:?pp}"
ALLOW_FAIL="${ALLOW_FAIL:-0}"

export RUN_ID
export MASTER_PORT="${MASTER_PORT:-29500}"
# Caller may set CUDA_VISIBLE_DEVICES (e.g. single-GPU runs); do not clear it.

if [[ "${MODEL_PRESET:-}" == "large" ]]; then
  load_large_model_config
  apply_model_preset large
elif [[ "${MODEL_PRESET:-}" == "original" ]]; then
  apply_model_preset original
fi

start_gpu_monitor
trap stop_gpu_monitor EXIT

megatron_common_args
EXTRA=()
if [[ -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE:-}" ]]; then
  EXTRA+=(--num-layers-per-virtual-pipeline-stage "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}")
fi

set +e
run_megatron "${NPROC}" \
  "${MEG_ARGS[@]}" \
  --tensor-model-parallel-size "${TP}" \
  --pipeline-model-parallel-size "${PP}" \
  "${EXTRA[@]}"
RC=$?
set -e

if [[ "${ALLOW_FAIL}" == "1" ]]; then
  exit 0
fi
exit "${RC}"
