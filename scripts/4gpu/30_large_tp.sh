#!/usr/bin/env bash
# Large-model performance benchmarks (after capacity freeze).
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S4="${LAB_ROOT}/scripts/4gpu"
export EXPERIMENT=4gpu MODEL_PRESET=large
export TRAIN_ITERS="${TRAIN_ITERS:-25}" WARMUP_ITERS="${WARMUP_ITERS:-5}"
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
unset CUDA_VISIBLE_DEVICES || true

echo "=== C: TP2 baseline ==="
MASTER_PORT=29801 bash "${S4}/run_megatron_job.sh" large_tp2 2 2 1

echo "=== D1: TP2xDP2 strong (GBS same) ==="
MASTER_PORT=29802 bash "${S4}/run_megatron_job.sh" large_tp2dp2_strong 4 2 1

echo "=== D2: TP2xDP2 weak (GBS*2) ==="
GLOBAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE * 2)) MASTER_PORT=29803 \
  bash "${S4}/run_megatron_job.sh" large_tp2dp2_weak 4 2 1
