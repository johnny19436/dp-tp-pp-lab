#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${LAB_ROOT}/scripts/common.sh"

export RUN_ID="tp2"
unset CUDA_VISIBLE_DEVICES || true
export MASTER_PORT=29503

start_gpu_monitor
trap stop_gpu_monitor EXIT

megatron_common_args
# Pure TP: no sequence parallel / overlap (plan §10)
run_megatron 2 \
  "${MEG_ARGS[@]}" \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1
