#!/usr/bin/env bash
# Shared helpers for DP/TP/PP lab (2-GPU and 4-GPU experiments)
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEGATRON_DIR="${LAB_ROOT}/third_party/Megatron-LM"
NCCL_TESTS_DIR="${LAB_ROOT}/third_party/nccl-tests"

# Experiment namespace: unset / "2gpu" -> legacy dirs; "4gpu" -> */4gpu/
EXPERIMENT="${EXPERIMENT:-}"
if [[ "${EXPERIMENT}" == "4gpu" ]]; then
  LOG_DIR="${LAB_ROOT}/logs/4gpu"
  PROF_DIR="${LAB_ROOT}/profiles/4gpu"
  RESULT_DIR="${LAB_ROOT}/results/4gpu"
else
  LOG_DIR="${LAB_ROOT}/logs"
  PROF_DIR="${LAB_ROOT}/profiles"
  RESULT_DIR="${LAB_ROOT}/results"
fi
CACHE_DIR="${LAB_ROOT}/data_cache"

mkdir -p "${LOG_DIR}" "${PROF_DIR}" "${RESULT_DIR}" "${CACHE_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# Default = original (~1.3B) model from 2-GPU experiment
NUM_LAYERS="${NUM_LAYERS:-24}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-8192}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
# 4gpu experiment uses short iters (throughput estimate, not precision).
# Legacy 2gpu default remains 120/20 unless EXPERIMENT=4gpu.
if [[ "${EXPERIMENT}" == "4gpu" ]]; then
  TRAIN_ITERS="${TRAIN_ITERS:-25}"
  WARMUP_ITERS="${WARMUP_ITERS:-5}"
else
  TRAIN_ITERS="${TRAIN_ITERS:-120}"
  WARMUP_ITERS="${WARMUP_ITERS:-20}"
fi
LR="${LR:-1.5e-4}"
MIN_LR="${MIN_LR:-1.5e-5}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODEL_ID="${MODEL_ID:-original}"

# Apply named model preset (does not override vars already exported by caller
# unless FORCE_MODEL_PRESET=1).
apply_model_preset() {
  local preset="${1:-original}"
  MODEL_ID="${preset}"
  case "${preset}" in
    original)
      NUM_LAYERS=24
      HIDDEN_SIZE=2048
      FFN_HIDDEN_SIZE=8192
      NUM_ATTENTION_HEADS=16
      SEQ_LENGTH=2048
      VOCAB_SIZE=32000
      ;;
    large)
      # Target ~2B GPT; may be adjusted after capacity smokes (see results/4gpu/large_model_config.txt)
      NUM_LAYERS="${LARGE_NUM_LAYERS:-20}"
      HIDDEN_SIZE="${LARGE_HIDDEN_SIZE:-3072}"
      FFN_HIDDEN_SIZE="${LARGE_FFN_HIDDEN_SIZE:-12288}"
      NUM_ATTENTION_HEADS="${LARGE_NUM_ATTENTION_HEADS:-24}"
      SEQ_LENGTH="${LARGE_SEQ_LENGTH:-2048}"
      VOCAB_SIZE="${LARGE_VOCAB_SIZE:-32000}"
      ;;
    *)
      echo "Unknown model preset: ${preset}" >&2
      return 1
      ;;
  esac
}

# Load frozen large-model overrides if present
load_large_model_config() {
  local cfg="${RESULT_DIR}/large_model_config.env"
  if [[ -f "${cfg}" ]]; then
    # shellcheck disable=SC1090
    source "${cfg}"
  fi
}

megatron_common_args() {
  MEG_ARGS=(
    --use-mcore-models
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${SEQ_LENGTH}"
    --vocab-size "${VOCAB_SIZE}"
    --tokenizer-type NullTokenizer
    --mock-data
    --data-cache-path "${CACHE_DIR}"
    --split 99,1,0
    --no-create-attention-mask-in-dataloader
    --num-workers 2
    --bf16
    --attention-backend auto
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --init-method-std 0.02
    --seed 1234
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --lr "${LR}"
    --min-lr "${MIN_LR}"
    --lr-decay-style cosine
    --lr-warmup-iters "${WARMUP_ITERS}"
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --log-interval 1
    --log-throughput
    --eval-iters 0
    --eval-interval "${TRAIN_ITERS}"
    --save-interval 100000
  )
}

run_megatron() {
  local nproc="$1"
  shift
  local logfile="${LOG_DIR}/${RUN_ID}.log"
  echo "[$(date -u +%FT%TZ)] START ${RUN_ID} nproc=${nproc} model=${MODEL_ID} layers=${NUM_LAYERS} hidden=${HIDDEN_SIZE}" | tee "${logfile}"
  echo "CMD: python3 -m torch.distributed.run --standalone --nproc_per_node=${nproc} --master_port=${MASTER_PORT} pretrain_gpt.py $*" | tee -a "${logfile}"
  (
    cd "${MEGATRON_DIR}"
    python3 -m torch.distributed.run --standalone --nproc_per_node="${nproc}" --master_port="${MASTER_PORT}" \
      pretrain_gpt.py "$@"
  ) 2>&1 | tee -a "${logfile}"
  local rc=${PIPESTATUS[0]}
  echo "[$(date -u +%FT%TZ)] END ${RUN_ID} rc=${rc}" | tee -a "${logfile}"
  return "${rc}"
}

# Like run_megatron but does not fail the caller on non-zero (for OOM capacity tests)
run_megatron_allow_fail() {
  set +e
  run_megatron "$@"
  local rc=$?
  set -e
  return "${rc}"
}

start_gpu_monitor() {
  local monfile="${LOG_DIR}/${RUN_ID}.dmon.csv"
  nvidia-smi dmon -s pucvmet -d 1 -o DT -f "${monfile}" &
  MONITOR_PID=$!
  echo "${MONITOR_PID}" > "${LOG_DIR}/${RUN_ID}.dmon.pid"
  echo "GPU monitor pid=${MONITOR_PID} -> ${monfile}"
}

stop_gpu_monitor() {
  if [[ -f "${LOG_DIR}/${RUN_ID}.dmon.pid" ]]; then
    local pid
    pid="$(cat "${LOG_DIR}/${RUN_ID}.dmon.pid")"
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    rm -f "${LOG_DIR}/${RUN_ID}.dmon.pid"
  fi
}

param_count_approx() {
  # Rough GPT param count for logging (untied emb+out, SwiGLU FFN ≈ 3*h*ffn/2 if gated; Megatron SwiGLU uses 2*ffn intermediate)
  python3 - <<PY
L=${NUM_LAYERS}; H=${HIDDEN_SIZE}; F=${FFN_HIDDEN_SIZE}; V=${VOCAB_SIZE}; heads=${NUM_ATTENTION_HEADS}
# Attention: QKV + proj ≈ 4 * H * H
# SwiGLU MLP in Megatron: gate+up = 2 * H * F, down = F * H → 3 * H * F
# Layer norms negligible; untied emb + out: 2 * V * H
attn = 4 * H * H
mlp = 3 * H * F
per_layer = attn + mlp
emb = 2 * V * H
total = L * per_layer + emb
print(f"approx_params={total} ({total/1e9:.3f}B)")
print(f"layers={L} hidden={H} ffn={F} heads={heads} vocab={V}")
PY
}
