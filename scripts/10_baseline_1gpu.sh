#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${LAB_ROOT}/scripts/common.sh"

export RUN_ID="baseline_1gpu"
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29501

start_gpu_monitor
trap stop_gpu_monitor EXIT

megatron_common_args
run_megatron 1 \
  "${MEG_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1
