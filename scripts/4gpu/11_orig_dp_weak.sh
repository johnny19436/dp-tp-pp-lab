#!/usr/bin/env bash
# Original-model DP weak scaling: GBS = 64 * DP
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S4="${LAB_ROOT}/scripts/4gpu"
export EXPERIMENT=4gpu MODEL_PRESET=original MICRO_BATCH_SIZE=1
export TRAIN_ITERS="${TRAIN_ITERS:-25}" WARMUP_ITERS="${WARMUP_ITERS:-5}"

# 1GPU already covered by orig_1gpu in strong; re-run weak variants for DP2/DP4
export GLOBAL_BATCH_SIZE=128
MASTER_PORT=29611 bash "${S4}/run_megatron_job.sh" orig_dp2_weak 2 1 1
export GLOBAL_BATCH_SIZE=256
MASTER_PORT=29612 bash "${S4}/run_megatron_job.sh" orig_dp4_weak 4 1 1
