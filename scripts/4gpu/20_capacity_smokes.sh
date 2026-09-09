#!/usr/bin/env bash
# Capacity smoke tests for large model (short iters). Writes capacity CSV rows via logs.
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S4="${LAB_ROOT}/scripts/4gpu"
export EXPERIMENT=4gpu MODEL_PRESET=large
export TRAIN_ITERS=5 WARMUP_ITERS=1
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" MICRO_BATCH_SIZE=1
export ALLOW_FAIL=1

source "${LAB_ROOT}/scripts/common.sh"
load_large_model_config
apply_model_preset large
param_count_approx | tee "${RESULT_DIR}/large_model_params.txt"

echo "=== Capacity B1: 1GPU (expect OOM) ==="
MASTER_PORT=29701 CUDA_VISIBLE_DEVICES=0 ALLOW_FAIL=1 \
  bash "${S4}/run_megatron_job.sh" cap_1gpu 1 1 1 || true

echo "=== Capacity B2: DP4 (expect OOM if replica OOMs) ==="
unset CUDA_VISIBLE_DEVICES
MASTER_PORT=29702 ALLOW_FAIL=1 \
  bash "${S4}/run_megatron_job.sh" cap_dp4 4 1 1 || true

echo "=== Capacity B3: TP2 ==="
MASTER_PORT=29703 ALLOW_FAIL=1 \
  bash "${S4}/run_megatron_job.sh" cap_tp2 2 2 1 || true

echo "=== Capacity B4: PP2 ==="
MASTER_PORT=29704 ALLOW_FAIL=1 \
  bash "${S4}/run_megatron_job.sh" cap_pp2 2 1 2 || true

echo "Capacity smokes done. Inspect logs/4gpu/cap_*.log"
