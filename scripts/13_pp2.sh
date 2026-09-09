#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${LAB_ROOT}/scripts/common.sh"

export RUN_ID="pp2_mbs${MICRO_BATCH_SIZE}"
unset CUDA_VISIBLE_DEVICES || true
export MASTER_PORT="${MASTER_PORT:-29504}"

start_gpu_monitor
trap stop_gpu_monitor EXIT

megatron_common_args
run_megatron 2 \
  "${MEG_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 2
