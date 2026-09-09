#!/usr/bin/env bash
# Short Nsight profiles for required 4-GPU configs.
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export EXPERIMENT=4gpu
source "${LAB_ROOT}/scripts/common.sh"

MODE="${1:?usage: profile_nsys_4gpu.sh <orig_dp4|large_tp2|large_tp2dp2|large_pp2|large_pp2dp2|large_tp4|large_pp4|large_tp2pp2>}"
export TRAIN_ITERS=15
export WARMUP_ITERS=5
export MASTER_PORT=29900
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
unset CUDA_VISIBLE_DEVICES || true

case "${MODE}" in
  orig_dp4)
    apply_model_preset original
    export RUN_ID="nsys_orig_dp4"; TP=1; PP=1; NPROC=4
    ;;
  large_tp2)
    load_large_model_config; apply_model_preset large
    export RUN_ID="nsys_large_tp2"; TP=2; PP=1; NPROC=2
    ;;
  large_tp2dp2)
    load_large_model_config; apply_model_preset large
    export RUN_ID="nsys_large_tp2dp2"; TP=2; PP=1; NPROC=4
    ;;
  large_pp2)
    load_large_model_config; apply_model_preset large
    export RUN_ID="nsys_large_pp2"; TP=1; PP=2; NPROC=2
    ;;
  large_pp2dp2)
    load_large_model_config; apply_model_preset large
    export RUN_ID="nsys_large_pp2dp2"; TP=1; PP=2; NPROC=4
    ;;
  large_tp4)
    load_large_model_config; apply_model_preset large
    export RUN_ID="nsys_large_tp4"; TP=4; PP=1; NPROC=4
    ;;
  large_pp4)
    load_large_model_config; apply_model_preset large
    export RUN_ID="nsys_large_pp4"; TP=1; PP=4; NPROC=4
    ;;
  large_tp2pp2)
    load_large_model_config; apply_model_preset large
    export RUN_ID="nsys_large_tp2pp2"; TP=2; PP=2; NPROC=4
    ;;
  *)
    echo "unknown mode ${MODE}"; exit 1
    ;;
esac

megatron_common_args
OUT="${PROF_DIR}/${RUN_ID}"
cd "${MEGATRON_DIR}"
nsys profile \
  --force-overwrite=true \
  -o "${OUT}" \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --duration=35 \
  python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    pretrain_gpt.py \
    "${MEG_ARGS[@]}" \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    --profile \
    --profile-step-start 8 \
    --profile-step-end 12 \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}.log" || true

nsys stats --report cuda_gpu_kern_sum --report nvtx_sum "${OUT}.nsys-rep" \
  > "${OUT}_stats.txt" || true
echo "Wrote ${OUT}.nsys-rep and ${OUT}_stats.txt"
