#!/usr/bin/env python3
"""Parse 4-GPU experiment logs into CSVs under results/4gpu/."""
from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
LOG_DIR = LAB / "logs" / "4gpu"
RESULT_DIR = LAB / "results" / "4gpu"
PROF_DIR = LAB / "profiles" / "4gpu"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)",
    re.IGNORECASE,
)
MEM_RE = re.compile(r"max allocated:\s*([\d.]+)", re.IGNORECASE)
OOM_RE = re.compile(r"out of memory|CUDA out of memory|torch.OutOfMemoryError", re.IGNORECASE)


def parse_dmon(path: Path, max_gpu: int = 4):
    utils = {i: [] for i in range(max_gpu)}
    mems = {i: [] for i in range(max_gpu)}
    pwrs = {i: [] for i in range(max_gpu)}
    if not path.exists():
        return utils, mems, pwrs
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            for i, p in enumerate(parts):
                if p in {str(g) for g in range(max_gpu)} and i >= 1:
                    gpu = int(p)
                    # columns after gpu: pwr gtemp mtemp sm mem ...
                    pwr = float(parts[i + 1])
                    sm = float(parts[i + 4])
                    mem_pct = float(parts[i + 5])
                    utils[gpu].append(sm)
                    mems[gpu].append(mem_pct)
                    pwrs[gpu].append(pwr)
                    break
        except (ValueError, IndexError):
            continue
    return utils, mems, pwrs


def avg(xs):
    return sum(xs) / len(xs) if xs else None


def peak(xs):
    return max(xs) if xs else None


def pct_to_mb(pct):
    return pct * 40960 / 100.0 if pct is not None else None


def parse_training_log(path: Path, warmup: int = 20) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    oom = bool(OOM_RE.search(text))
    step_ms = []
    peak_alloc = []
    for line in text.splitlines():
        m = ITER_RE.search(line)
        if m:
            it = int(m.group(1))
            ms = float(m.group(2))
            if it > warmup:
                step_ms.append(ms)
        m2 = MEM_RE.search(line)
        if m2:
            peak_alloc.append(float(m2.group(1)))
    out = {"oom": oom, "completed_iters": len(step_ms)}
    if step_ms:
        step_ms_sorted = sorted(step_ms)
        p95 = step_ms_sorted[min(len(step_ms_sorted) - 1, int(math.ceil(0.95 * len(step_ms_sorted)) - 1))]
        out.update(
            {
                "n_measured": len(step_ms),
                "mean_step_ms": statistics.mean(step_ms),
                "median_step_ms": statistics.median(step_ms),
                "p95_step_ms": p95,
                "peak_alloc_mb": max(peak_alloc) if peak_alloc else None,
            }
        )
    elif peak_alloc:
        out["peak_alloc_mb"] = max(peak_alloc)
    return out


# (run_id, config, model_id, num_gpus, tp, pp, dp, micro, global, seq, warmup)
TRAIN_CONFIGS = [
    # warmup=5 matches short 4gpu runs (TRAIN_ITERS=25)
    ("orig_1gpu", "orig_1GPU", "original", 1, 1, 1, 1, 1, 64, 2048, 5),
    ("orig_dp2", "orig_DP2", "original", 2, 1, 1, 2, 1, 64, 2048, 5),
    ("orig_dp4", "orig_DP4", "original", 4, 1, 1, 4, 1, 64, 2048, 5),
    ("orig_dp2_weak", "orig_DP2_weak", "original", 2, 1, 1, 2, 1, 128, 2048, 5),
    ("orig_dp4_weak", "orig_DP4_weak", "original", 4, 1, 1, 4, 1, 256, 2048, 5),
    ("large_tp2", "large_TP2", "large", 2, 2, 1, 1, 1, 64, 2048, 5),
    ("large_tp2dp2_strong", "large_TP2DP2_strong", "large", 4, 2, 1, 2, 1, 64, 2048, 5),
    ("large_tp2dp2_weak", "large_TP2DP2_weak", "large", 4, 2, 1, 2, 1, 128, 2048, 5),
    ("large_pp2", "large_PP2", "large", 2, 1, 2, 1, 1, 64, 2048, 5),
    ("large_pp2dp2_strong", "large_PP2DP2_strong", "large", 4, 1, 2, 2, 1, 64, 2048, 5),
    ("large_pp2dp2_weak", "large_PP2DP2_weak", "large", 4, 1, 2, 2, 1, 128, 2048, 5),
    ("large_tp4", "large_TP4", "large", 4, 4, 1, 1, 1, 64, 2048, 5),
    ("large_pp4_mbs1", "large_PP4_mbs1", "large", 4, 1, 4, 1, 1, 64, 2048, 5),
    ("large_pp4_mbs2", "large_PP4_mbs2", "large", 4, 1, 4, 1, 2, 64, 2048, 5),
    ("large_pp4_mbs4", "large_PP4_mbs4", "large", 4, 1, 4, 1, 4, 64, 2048, 5),
    ("large_tp2pp2", "large_TP2PP2", "large", 4, 2, 2, 1, 1, 64, 2048, 5),
]

CAP_CONFIGS = [
    ("cap_1gpu", "cap_1GPU", 1, 1, 1, 1),
    ("cap_dp4", "cap_DP4", 4, 1, 1, 4),
    ("cap_tp2", "cap_TP2", 2, 2, 1, 1),
    ("cap_pp2", "cap_PP2", 2, 1, 2, 1),
]


def dmon_row(run_id: str, ng: int) -> dict:
    utils, mems, pwrs = parse_dmon(LOG_DIR / f"{run_id}.dmon.csv")
    row = {}
    for g in range(4):
        row[f"gpu{g}_avg_util"] = round(avg(utils[g]), 1) if utils[g] else ""
        row[f"gpu{g}_peak_mem_mb"] = round(pct_to_mb(peak(mems[g])), 1) if mems[g] else ""
        row[f"gpu{g}_avg_power"] = round(avg(pwrs[g]), 1) if pwrs[g] else ""
    return row


def write_training_csv():
    rows = []
    for run_id, config, model_id, ng, tp, pp, dp, mbs, gbs, seq, warmup in TRAIN_CONFIGS:
        log = LOG_DIR / f"{run_id}.log"
        if not log.exists():
            print(f"WARN: missing {log}")
            continue
        metrics = parse_training_log(log, warmup=warmup)
        if not metrics.get("mean_step_ms"):
            print(f"WARN: no measured iters in {log} (oom={metrics.get('oom')})")
            continue
        tokens_per_step = gbs * seq
        mean_s = metrics["mean_step_ms"] / 1000.0
        tps = tokens_per_step / mean_s
        row = {
            "run_id": run_id,
            "config": config,
            "model_id": model_id,
            "num_gpus": ng,
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "micro_batch": mbs,
            "global_batch": gbs,
            "seq_len": seq,
            "mean_step_ms": round(metrics["mean_step_ms"], 3),
            "median_step_ms": round(metrics["median_step_ms"], 3),
            "p95_step_ms": round(metrics["p95_step_ms"], 3),
            "tokens_per_sec": round(tps, 1),
            "tokens_per_sec_per_gpu": round(tps / ng, 1),
            "peak_alloc_mb": round(metrics["peak_alloc_mb"], 2) if metrics.get("peak_alloc_mb") else "",
            "experiment_generation": "4gpu",
        }
        row.update(dmon_row(run_id, ng))
        rows.append(row)

    out = RESULT_DIR / "training_results_4gpu.csv"
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out} ({len(rows)} rows)")
    else:
        print("No training rows")
    return rows


def write_capacity_csv():
    rows = []
    for run_id, config, ng, tp, pp, dp in CAP_CONFIGS:
        log = LOG_DIR / f"{run_id}.log"
        if not log.exists():
            continue
        metrics = parse_training_log(log, warmup=0)
        fits = (not metrics.get("oom", False)) and metrics.get("completed_iters", 0) > 0
        # Also treat explicit END rc!=0 with OOM
        text = log.read_text(errors="ignore")
        if OOM_RE.search(text):
            fits = False
            status = "OOM"
        elif fits:
            status = "FITS"
        else:
            status = "FAIL"
        row = {
            "run_id": run_id,
            "config": config,
            "num_gpus": ng,
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "status": status,
            "fits": fits,
            "peak_alloc_mb": round(metrics["peak_alloc_mb"], 2) if metrics.get("peak_alloc_mb") else "",
            "notes": "smoke TRAIN_ITERS=5",
        }
        rows.append(row)
    out = RESULT_DIR / "capacity_results_4gpu.csv"
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out}")
    return rows


def parse_nsys_nccl_share(stats_path: Path) -> float | None:
    """Estimate % of CUDA kernel time in NCCL from nsys cuda_gpu_kern_sum."""
    if not stats_path.exists():
        return None
    text = stats_path.read_text(errors="ignore")
    # Look for Time(%) column rows; sum nccl* percentages
    pct_re = re.compile(r"^\s*([\d.]+)\s+.*?nccl", re.IGNORECASE | re.MULTILINE)
    # Also lines where kernel name contains nccl
    total_nccl = 0.0
    for line in text.splitlines():
        if "nccl" not in line.lower():
            continue
        m = re.match(r"^\s*([\d.]+)\s+", line)
        if m:
            total_nccl += float(m.group(1))
    return round(total_nccl, 2) if total_nccl > 0 else None


def write_nsys_summary():
    modes = [
        "nsys_orig_dp4",
        "nsys_large_tp2",
        "nsys_large_tp2dp2",
        "nsys_large_pp2",
        "nsys_large_pp2dp2",
        "nsys_large_tp4",
        "nsys_large_pp4",
        "nsys_large_tp2pp2",
    ]
    rows = []
    for m in modes:
        share = parse_nsys_nccl_share(PROF_DIR / f"{m}_stats.txt")
        rows.append({"profile": m, "nccl_kernel_time_pct_est": share if share is not None else ""})
    out = RESULT_DIR / "nsys_comm_summary_4gpu.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["profile", "nccl_kernel_time_pct_est"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


def main():
    write_training_csv()
    write_capacity_csv()
    write_nsys_summary()


if __name__ == "__main__":
    main()
