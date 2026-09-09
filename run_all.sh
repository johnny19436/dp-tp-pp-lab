#!/usr/bin/env bash
set -euo pipefail
LAB="$(cd "$(dirname "$0")" && pwd)"
cd "$LAB"
mkdir -p logs profiles results analysis data_cache
chmod +x scripts/*.sh scripts/*.py analysis/*.py || true

echo "==== 1. System check ===="
LAB="$LAB" bash scripts/00_system_check.sh

echo "==== 2. NCCL benchmark ===="
bash scripts/01_nccl_benchmark.sh

echo "==== 3. torch.distributed sanity ===="
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/02_torch_distributed_test.py | tee logs/02_torch_distributed_test.log

echo "==== 7. 1GPU baseline ===="
bash scripts/10_baseline_1gpu.sh

echo "==== 8. DP2 strong ===="
bash scripts/11_dp2.sh

echo "==== 9. DP2 weak ===="
bash scripts/14_dp_weak_scaling.sh

echo "==== 10. TP2 ===="
bash scripts/12_tp2.sh

echo "==== 11. PP2 mbs=1 ===="
MICRO_BATCH_SIZE=1 bash scripts/13_pp2.sh

echo "==== 12. PP mbs=2 ===="
MICRO_BATCH_SIZE=2 MASTER_PORT=29512 bash scripts/13_pp2.sh

echo "==== 12b. PP mbs=4 ===="
MICRO_BATCH_SIZE=4 MASTER_PORT=29514 bash scripts/13_pp2.sh

echo "==== 13-15. Nsight profiles ===="
bash scripts/profile_nsys.sh dp2 || true
bash scripts/profile_nsys.sh tp2 || true
bash scripts/profile_nsys.sh pp2 || true

echo "==== 16. Summarize ===="
python3 analysis/summarize.py

echo "ALL DONE at $(date -u)"
