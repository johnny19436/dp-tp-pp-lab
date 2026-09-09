#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${LAB_ROOT}/logs/00_system_check.log"
mkdir -p "${LAB_ROOT}/logs"
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
    print(i, torch.cuda.get_device_name(i))
print("NCCL:", torch.cuda.nccl.version())
PY
  echo "=== megatron ==="
  python3 - <<'PY'
import megatron.core as mc
print("megatron.core:", mc.__file__)
import subprocess, os
root=os.environ.get("LAB", "/root/working/dp-tp-pp-lab")
print(subprocess.check_output(["git","-C",f"{root}/third_party/Megatron-LM","rev-parse","HEAD"], text=True).strip())
PY
} | tee "${OUT}"
echo "Wrote ${OUT}"
