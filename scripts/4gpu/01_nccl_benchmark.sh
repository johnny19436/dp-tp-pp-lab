#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export EXPERIMENT=4gpu
source "${LAB_ROOT}/scripts/common.sh"

BUILD="${NCCL_TESTS_DIR}/build"
OUT_LOG="${LOG_DIR}/01_nccl_benchmark.log"
CSV="${RESULT_DIR}/nccl_results_4gpu.csv"
NGPU=4

{
  echo "=== $(date -u) NCCL microbenchmark ${NGPU} GPUs ==="
  for coll in all_reduce all_gather reduce_scatter; do
    echo "----- ${coll} -----"
    "${BUILD}/${coll}_perf" -b 8M -e 1G -f 2 -g "${NGPU}"
  done
} | tee "${OUT_LOG}"

python3 - <<'PY' "${OUT_LOG}" "${CSV}" "${NGPU}"
import csv, re, sys
log_path, csv_path, ngpu = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = open(log_path).read().splitlines()
rows = []
current = None
pat = re.compile(
    r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+"
)
for line in text:
    if "----- all_reduce" in line or ("all_reduce_perf" in line and "starting" in line):
        current = "all_reduce"
    elif "----- all_gather" in line or ("all_gather_perf" in line and "starting" in line):
        current = "all_gather"
    elif "----- reduce_scatter" in line or ("reduce_scatter_perf" in line and "starting" in line):
        current = "reduce_scatter"
    if current is None:
        continue
    m = pat.match(line)
    if not m:
        continue
    rows.append({
        "collective": current,
        "num_gpus": ngpu,
        "size_bytes": int(m.group(1)),
        "time_us": float(m.group(2)),
        "algbw_GBs": float(m.group(3)),
        "busbw_GBs": float(m.group(4)),
        "mode": "out_of_place",
    })
fields = ["collective","num_gpus","size_bytes","time_us","algbw_GBs","busbw_GBs","mode"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {len(rows)} rows -> {csv_path}")
PY
