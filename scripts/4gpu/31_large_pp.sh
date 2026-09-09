#!/usr/bin/env bash
# Large-model PP2 and PP2xDP2
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S4="${LAB_ROOT}/scripts/4gpu"
export EXPERIMENT=4gpu MODEL_PRESET=large
export TRAIN_ITERS="${TRAIN_ITERS:-25}" WARMUP_ITERS="${WARMUP_ITERS:-5}"
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
unset CUDA_VISIBLE_DEVICES || true

echo "=== E: PP2 baseline ==="
MASTER_PORT=29811 bash "${S4}/run_megatron_job.sh" large_pp2 2 1 2

echo "=== F1: PP2xDP2 strong ==="
MASTER_PORT=29812 bash "${S4}/run_megatron_job.sh" large_pp2dp2_strong 4 1 2

echo "=== F2: PP2xDP2 weak ==="
GLOBAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE * 2)) MASTER_PORT=29813 \
  bash "${S4}/run_megatron_job.sh" large_pp2dp2_weak 4 1 2
