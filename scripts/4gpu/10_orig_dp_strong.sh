#!/usr/bin/env bash
# Original-model DP strong scaling: 1 / 2 / 4 GPUs, GBS=64
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S4="${LAB_ROOT}/scripts/4gpu"
export EXPERIMENT=4gpu MODEL_PRESET=original
export GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=1
export TRAIN_ITERS="${TRAIN_ITERS:-25}" WARMUP_ITERS="${WARMUP_ITERS:-5}"

MASTER_PORT=29601 CUDA_VISIBLE_DEVICES=0 bash "${S4}/run_megatron_job.sh" orig_1gpu 1 1 1
# clear CUDA_VISIBLE_DEVICES for multi-GPU
unset CUDA_VISIBLE_DEVICES
MASTER_PORT=29602 bash "${S4}/run_megatron_job.sh" orig_dp2 2 1 1
MASTER_PORT=29603 bash "${S4}/run_megatron_job.sh" orig_dp4 4 1 1
