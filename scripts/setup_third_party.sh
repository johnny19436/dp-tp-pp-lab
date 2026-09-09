#!/usr/bin/env bash
# Fetch pinned third_party deps for this lab (not vendored in git).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="${ROOT}/third_party"
mkdir -p "${TP}"

MEGATRON_URL="${MEGATRON_URL:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_REF="${MEGATRON_REF:-be85fc5df550f2236a850ecbf0dd1bf1b4cb4814}"
NCCL_TESTS_URL="${NCCL_TESTS_URL:-https://github.com/NVIDIA/nccl-tests.git}"
NCCL_TESTS_REF="${NCCL_TESTS_REF:-master}"

echo "==> Megatron-LM @ ${MEGATRON_REF}"
if [[ ! -d "${TP}/Megatron-LM/.git" ]]; then
  git clone --filter=blob:none "${MEGATRON_URL}" "${TP}/Megatron-LM"
fi
git -C "${TP}/Megatron-LM" fetch --depth 1 origin "${MEGATRON_REF}" 2>/dev/null \
  || git -C "${TP}/Megatron-LM" fetch origin "${MEGATRON_REF}"
git -C "${TP}/Megatron-LM" checkout "${MEGATRON_REF}"
# Install editable without pulling a different torch (critical on NGC images)
MAX_JOBS="${MAX_JOBS:-8}" pip install -e "${TP}/Megatron-LM" --no-deps --no-build-isolation
pip install -q sentencepiece tiktoken 2>/dev/null || true

echo "==> nccl-tests"
if [[ ! -d "${TP}/nccl-tests/.git" ]]; then
  git clone --depth 1 "${NCCL_TESTS_URL}" "${TP}/nccl-tests"
fi
make -C "${TP}/nccl-tests" -j"$(nproc)" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" NCCL_HOME="${NCCL_HOME:-/usr}"

echo "==> done"
python3 - <<'PY'
import torch, megatron.core
print("torch", torch.__version__, "gpus", torch.cuda.device_count())
print("megatron", megatron.core.__file__)
PY
