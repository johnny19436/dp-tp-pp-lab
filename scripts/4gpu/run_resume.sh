#!/usr/bin/env bash
# Resume after PP4 mbs4 OOM abort: TP2xPP2 + Nsight + report
set -euo pipefail
LAB="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$LAB"
export EXPERIMENT=4gpu TRAIN_ITERS=25 WARMUP_ITERS=5
LOGDIR=logs/4gpu
exec > >(tee -a "$LOGDIR/run_resume.log") 2>&1
echo "==== RESUME START $(date -u) pid=$$ ===="

S4=scripts/4gpu
if ! grep -q 'END large_tp2pp2 rc=0' "$LOGDIR/large_tp2pp2.log" 2>/dev/null; then
  MICRO_BATCH_SIZE=1 MASTER_PORT=29825 MODEL_PRESET=large \
    bash "$S4/run_megatron_job.sh" large_tp2pp2 4 2 2
else
  echo "[skip] large_tp2pp2"
fi

echo "==== Nsight ===="
for m in orig_dp4 large_tp2 large_tp2dp2 large_pp2 large_pp2dp2 large_tp4 large_pp4 large_tp2pp2; do
  MASTER_PORT=$((29900 + RANDOM % 100)) bash "$S4/profile_nsys_4gpu.sh" "$m" || true
done

echo "==== Summarize + report ===="
python3 analysis/4gpu/summarize_4gpu.py
python3 analysis/4gpu/generate_report_4gpu.py
echo "ALL_4GPU_DONE $(date -u)"
ls -lh report_4gpu.pdf reports/4gpu/report_4gpu.pdf
