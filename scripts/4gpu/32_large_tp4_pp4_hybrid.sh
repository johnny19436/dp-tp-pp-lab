#!/usr/bin/env bash
# Large-model TP4, PP4 microbatch sweep, TP2xPP2
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S4="${LAB_ROOT}/scripts/4gpu"
export EXPERIMENT=4gpu MODEL_PRESET=large
export TRAIN_ITERS="${TRAIN_ITERS:-25}" WARMUP_ITERS="${WARMUP_ITERS:-5}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
unset CUDA_VISIBLE_DEVICES || true

echo "=== G: TP4 ==="
MICRO_BATCH_SIZE=1 MASTER_PORT=29821 bash "${S4}/run_megatron_job.sh" large_tp4 4 4 1

echo "=== H: PP4 mbs=1 ==="
MICRO_BATCH_SIZE=1 MASTER_PORT=29822 bash "${S4}/run_megatron_job.sh" large_pp4_mbs1 4 1 4

echo "=== H: PP4 mbs=2 ==="
MICRO_BATCH_SIZE=2 MASTER_PORT=29823 bash "${S4}/run_megatron_job.sh" large_pp4_mbs2 4 1 4

echo "=== H: PP4 mbs=4 (may OOM; continue anyway) ==="
MICRO_BATCH_SIZE=4 MASTER_PORT=29824 ALLOW_FAIL=1 bash "${S4}/run_megatron_job.sh" large_pp4_mbs4 4 1 4 || true

echo "=== I: TP2xPP2 ==="
MICRO_BATCH_SIZE=1 MASTER_PORT=29825 bash "${S4}/run_megatron_job.sh" large_tp2pp2 4 2 2
