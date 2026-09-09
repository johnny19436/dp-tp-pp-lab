#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${LAB_ROOT}/scripts/common.sh"

export RUN_ID="smoke_dp2"
export NUM_LAYERS=2
export HIDDEN_SIZE=256
export FFN_HIDDEN_SIZE=1024
export NUM_ATTENTION_HEADS=4
export SEQ_LENGTH=512
export VOCAB_SIZE=1024
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=4
export TRAIN_ITERS=5
export WARMUP_ITERS=1
export MASTER_PORT=29611

megatron_common_args
run_megatron 2 \
  "${MEG_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1

echo "SMOKE OK"
