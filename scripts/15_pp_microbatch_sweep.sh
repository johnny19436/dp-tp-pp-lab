#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for mbs in 1 2 4; do
  echo "===== PP microbatch sweep mbs=${mbs} ====="
  MICRO_BATCH_SIZE="${mbs}" MASTER_PORT=$((29510 + mbs)) \
    bash "${LAB_ROOT}/scripts/13_pp2.sh"
done
