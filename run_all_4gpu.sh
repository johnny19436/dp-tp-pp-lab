#!/usr/bin/env bash
# End-to-end 4-GPU experiment runner (plan_4gpu.txt execution order).
# Short iters: TRAIN_ITERS=25, WARMUP_ITERS=5 (~20 measured steps).
set -euo pipefail
LAB="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB"
export EXPERIMENT=4gpu
export TRAIN_ITERS="${TRAIN_ITERS:-25}"
export WARMUP_ITERS="${WARMUP_ITERS:-5}"
mkdir -p logs/4gpu profiles/4gpu results/4gpu analysis/4gpu/figures reports/4gpu
chmod +x scripts/4gpu/*.sh scripts/4gpu/*.py analysis/4gpu/*.py 2>/dev/null || true
S4=scripts/4gpu

echo "==== Phase 1: System check ===="
bash "${S4}/00_system_check.sh"

echo "==== Phase 2: NCCL 4-GPU ===="
bash "${S4}/01_nccl_benchmark.sh"

echo "==== Phase 2b: torch.distributed 4-GPU ===="
python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${S4}/02_torch_distributed_test.py" | tee logs/4gpu/02_torch_distributed_test.log

echo "==== Phase 3: Original model DP strong 1/2/4 ===="
bash "${S4}/10_orig_dp_strong.sh"

echo "==== Phase 3b: Original model DP weak ===="
bash "${S4}/11_orig_dp_weak.sh"

echo "==== Phase 4: Freeze large model ===="
bash "${S4}/21_freeze_large_model.sh"

echo "==== Phase 5: Capacity smokes ===="
bash "${S4}/20_capacity_smokes.sh"

python3 analysis/4gpu/adjust_large_model.py || true
if [[ -f results/4gpu/NEED_RECAPACITY ]]; then
  echo "==== Re-freeze + re-capacity after size adjust ===="
  # shellcheck disable=SC1091
  source results/4gpu/large_model_config.env
  bash "${S4}/21_freeze_large_model.sh"
  bash "${S4}/20_capacity_smokes.sh"
  rm -f results/4gpu/NEED_RECAPACITY
fi

echo "==== Phase 6-7: Large TP2 / TP2xDP2 ===="
bash "${S4}/30_large_tp.sh"

echo "==== Phase 8-9: Large PP2 / PP2xDP2 ===="
bash "${S4}/31_large_pp.sh"

echo "==== Phase 10-12: TP4 / PP4 / TP2xPP2 ===="
bash "${S4}/32_large_tp4_pp4_hybrid.sh"

echo "==== Phase 13: Nsight profiles ===="
for m in orig_dp4 large_tp2 large_tp2dp2 large_pp2 large_pp2dp2 large_tp4 large_pp4 large_tp2pp2; do
  MASTER_PORT=$((29900 + RANDOM % 100)) bash "${S4}/profile_nsys_4gpu.sh" "$m" || true
done

echo "==== Phase 14: Summarize ===="
python3 analysis/4gpu/summarize_4gpu.py

echo "==== Phase 15: Report ===="
python3 analysis/4gpu/generate_report_4gpu.py

echo "ALL 4GPU DONE at $(date -u)"
ls -lh reports/4gpu/report_4gpu.pdf report_4gpu.pdf 2>/dev/null || true
