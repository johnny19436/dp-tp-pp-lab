#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${LAB_ROOT}/scripts/common.sh"

export RUN_ID="dp2_weak"
export GLOBAL_BATCH_SIZE=128
unset CUDA_VISIBLE_DEVICES || true
export MASTER_PORT=29505

start_gpu_monitor
trap stop_gpu_monitor EXIT

megatron_common_args
run_megatron 2 \
  "${MEG_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1
