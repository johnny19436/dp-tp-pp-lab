#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${1:-manual}"
nvidia-smi dmon -s pucvmet -d 1 -o DT -f "${LAB_ROOT}/logs/${RUN_ID}.dmon.csv"
