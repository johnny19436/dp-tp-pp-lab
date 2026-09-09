#!/usr/bin/env bash
# Freeze large-model config after optional capacity-driven adjustment.
# Default plan target; override via LARGE_* env vars before calling.
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export EXPERIMENT=4gpu
source "${LAB_ROOT}/scripts/common.sh"

LARGE_NUM_LAYERS="${LARGE_NUM_LAYERS:-20}"
LARGE_HIDDEN_SIZE="${LARGE_HIDDEN_SIZE:-3072}"
LARGE_FFN_HIDDEN_SIZE="${LARGE_FFN_HIDDEN_SIZE:-12288}"
LARGE_NUM_ATTENTION_HEADS="${LARGE_NUM_ATTENTION_HEADS:-24}"
LARGE_SEQ_LENGTH="${LARGE_SEQ_LENGTH:-2048}"
LARGE_VOCAB_SIZE="${LARGE_VOCAB_SIZE:-32000}"
LARGE_GLOBAL_BATCH_SIZE="${LARGE_GLOBAL_BATCH_SIZE:-64}"

CFG_ENV="${RESULT_DIR}/large_model_config.env"
CFG_TXT="${RESULT_DIR}/large_model_config.txt"

cat > "${CFG_ENV}" <<EOF
# Frozen large-model config for 4-GPU experiment
export LARGE_NUM_LAYERS=${LARGE_NUM_LAYERS}
export LARGE_HIDDEN_SIZE=${LARGE_HIDDEN_SIZE}
export LARGE_FFN_HIDDEN_SIZE=${LARGE_FFN_HIDDEN_SIZE}
export LARGE_NUM_ATTENTION_HEADS=${LARGE_NUM_ATTENTION_HEADS}
export LARGE_SEQ_LENGTH=${LARGE_SEQ_LENGTH}
export LARGE_VOCAB_SIZE=${LARGE_VOCAB_SIZE}
export LARGE_GLOBAL_BATCH_SIZE=${LARGE_GLOBAL_BATCH_SIZE}
EOF

apply_model_preset large
{
  echo "Frozen at $(date -u)"
  echo "Source: plan_4gpu.txt §6 (adjusted only if capacity tests require)"
  param_count_approx
  cat "${CFG_ENV}"
} | tee "${CFG_TXT}"

echo "Wrote ${CFG_ENV} and ${CFG_TXT}"
