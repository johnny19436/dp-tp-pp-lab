#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${LAB_ROOT}/scripts/common.sh"

MODE="${1:?usage: profile_nsys.sh <dp2|tp2|pp2>}"
export TRAIN_ITERS=30
export WARMUP_ITERS=10
export MASTER_PORT=29700
unset CUDA_VISIBLE_DEVICES || true

case "${MODE}" in
  dp2) export RUN_ID="nsys_dp2"; TP=1; PP=1; NPROC=2 ;;
  tp2) export RUN_ID="nsys_tp2"; TP=2; PP=1; NPROC=2 ;;
  pp2) export RUN_ID="nsys_pp2"; TP=1; PP=2; NPROC=2 ;;
  *) echo "unknown mode ${MODE}"; exit 1 ;;
esac

megatron_common_args
OUT="${PROF_DIR}/${RUN_ID}"
cd "${MEGATRON_DIR}"
nsys profile \
  --force-overwrite=true \
  -o "${OUT}" \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --duration=45 \
  python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    pretrain_gpt.py \
    "${MEG_ARGS[@]}" \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --profile \
    --profile-step-start 15 \
    --profile-step-end 20 \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

nsys stats --report cuda_gpu_kern_sum --report nvtx_sum "${OUT}.nsys-rep" \
  > "${OUT}_stats.txt" || true
echo "Wrote ${OUT}.nsys-rep"
