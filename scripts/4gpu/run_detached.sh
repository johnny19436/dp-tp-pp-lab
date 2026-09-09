#!/usr/bin/env bash
# Detached 4-GPU suite (survives SSH disconnect). Skips jobs with END rc=0.
set -euo pipefail
LAB="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$LAB"
export EXPERIMENT=4gpu TRAIN_ITERS="${TRAIN_ITERS:-25}" WARMUP_ITERS="${WARMUP_ITERS:-5}"
LOGDIR=logs/4gpu
mkdir -p "$LOGDIR" profiles/4gpu results/4gpu analysis/4gpu/figures reports/4gpu
S4=scripts/4gpu
exec > >(tee -a "$LOGDIR/run_detached.log") 2>&1

done_ok() {
  local id="$1"
  [[ -f "$LOGDIR/${id}.log" ]] && grep -q "END ${id} rc=0" "$LOGDIR/${id}.log"
}

run_job() {
  local id="$1"; shift
  if done_ok "$id"; then
    echo "[skip] $id already OK"
    return 0
  fi
  echo "[run] $id $*"
  "$@"
}

echo "==== DETACHED START $(date -u) pid=$$ ===="

# Phase 3 strong
run_job orig_1gpu env MASTER_PORT=29601 CUDA_VISIBLE_DEVICES=0 MODEL_PRESET=original \
  GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=1 bash "$S4/run_megatron_job.sh" orig_1gpu 1 1 1
unset CUDA_VISIBLE_DEVICES || true
run_job orig_dp2 env MASTER_PORT=29602 MODEL_PRESET=original \
  GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=1 bash "$S4/run_megatron_job.sh" orig_dp2 2 1 1
run_job orig_dp4 env MASTER_PORT=29603 MODEL_PRESET=original \
  GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=1 bash "$S4/run_megatron_job.sh" orig_dp4 4 1 1

# Phase 3 weak
run_job orig_dp2_weak env MASTER_PORT=29611 MODEL_PRESET=original \
  GLOBAL_BATCH_SIZE=128 MICRO_BATCH_SIZE=1 bash "$S4/run_megatron_job.sh" orig_dp2_weak 2 1 1
run_job orig_dp4_weak env MASTER_PORT=29612 MODEL_PRESET=original \
  GLOBAL_BATCH_SIZE=256 MICRO_BATCH_SIZE=1 bash "$S4/run_megatron_job.sh" orig_dp4_weak 4 1 1

echo "==== Phase 4-5 freeze + capacity ===="
bash "$S4/21_freeze_large_model.sh"
bash "$S4/20_capacity_smokes.sh"
python3 analysis/4gpu/adjust_large_model.py || true
if [[ -f results/4gpu/NEED_RECAPACITY ]]; then
  # shellcheck disable=SC1091
  source results/4gpu/large_model_config.env
  bash "$S4/21_freeze_large_model.sh"
  bash "$S4/20_capacity_smokes.sh"
  rm -f results/4gpu/NEED_RECAPACITY
fi

echo "==== Phase 6-12 large model benches ===="
bash "$S4/30_large_tp.sh"
bash "$S4/31_large_pp.sh"
bash "$S4/32_large_tp4_pp4_hybrid.sh"

echo "==== Phase 13 Nsight ===="
for m in orig_dp4 large_tp2 large_tp2dp2 large_pp2 large_pp2dp2 large_tp4 large_pp4 large_tp2pp2; do
  MASTER_PORT=$((29900 + RANDOM % 100)) bash "$S4/profile_nsys_4gpu.sh" "$m" || true
done

echo "==== Phase 14-15 summarize + report ===="
python3 analysis/4gpu/summarize_4gpu.py
python3 analysis/4gpu/generate_report_4gpu.py
echo "ALL_4GPU_DONE $(date -u)"
ls -lh report_4gpu.pdf reports/4gpu/report_4gpu.pdf
