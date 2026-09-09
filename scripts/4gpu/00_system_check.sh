#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export EXPERIMENT=4gpu
source "${LAB_ROOT}/scripts/common.sh"
OUT="${LOG_DIR}/00_system_check.log"
{
  echo "=== timestamp ==="; date -u
  echo "=== hostname ==="; hostname
  echo "=== nvidia-smi ==="; nvidia-smi
  echo "=== nvidia-smi topo -m ==="; nvidia-smi topo -m
  echo "=== nvidia-smi nvlink -s ==="; nvidia-smi nvlink -s
  echo "=== nvcc ==="; nvcc --version || true
  echo "=== python ==="; python3 --version
  echo "=== pytorch ==="
  python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i),
          "mem_GB", round(torch.cuda.get_device_properties(i).total_memory/1024**3, 1))
print("NCCL:", torch.cuda.nccl.version())
PY
  echo "=== megatron ==="
  python3 - <<PY
import megatron.core as mc, subprocess
print("megatron.core:", mc.__file__)
print(subprocess.check_output(
    ["git","-C","${MEGATRON_DIR}","rev-parse","HEAD"], text=True).strip())
PY
} | tee "${OUT}"
echo "Wrote ${OUT}"
