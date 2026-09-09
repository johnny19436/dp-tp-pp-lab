#!/usr/bin/env python3
"""Parse Megatron training logs + dmon into results/training_results.csv and print tables."""
from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
LOG_DIR = LAB / "logs"
RESULT_DIR = LAB / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Megatron iteration line patterns (tolerant)
ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)",
    re.IGNORECASE,
)
THROUGHPUT_RE = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)", re.IGNORECASE)
# tokens/s sometimes logged differently; we compute from global batch * seq / step_time
MEM_RE = re.compile(r"max allocated:\s*([\d.]+)", re.IGNORECASE)


def parse_dmon(path: Path) -> tuple[float | None, float | None, float | None, float | None]:
    """Return gpu0_avg_util, gpu1_avg_util, gpu0_peak_mem_mb, gpu1_peak_mem_mb from dmon."""
    if not path.exists():
        return None, None, None, None
    utils = {0: [], 1: []}
    mems = {0: [], 1: []}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.startswith("#") or "gpu" in line.lower() and "idx" in line.lower():
            # header-ish
            parts = line.split()
            if parts and parts[0] in {"#", "Date", "date"}:
                continue
        parts = line.split()
        # dmon -o DT: Date Time gpu pwr gtemp mtemp sm mem enc dec mclk pclk
        # Example: 2026/... 07:..  0  50  24  24  0  0 ...
        if len(parts) < 8:
            continue
        try:
            # find gpu index: first small int 0/1 after time-like field
            gpu = None
            for i, p in enumerate(parts):
                if p in {"0", "1"} and i >= 1:
                    # heuristic: next fields numeric
                    gpu = int(p)
                    sm = float(parts[i + 4]) if len(parts) > i + 4 else float("nan")
                    # memory usage % is often index i+5; absolute MiB not always present
                    # With -s pucvmet: pwr util ... mem
                    # Actually columns: gpu pwr gtemp mtemp sm mem enc dec mclk pclk
                    sm = float(parts[i + 4])
                    mem_pct = float(parts[i + 5])
                    break
            else:
                continue
            if gpu in (0, 1) and not math.isnan(sm):
                utils[gpu].append(sm)
                mems[gpu].append(mem_pct)
        except (ValueError, IndexError):
            continue

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    def peak(xs):
        return max(xs) if xs else None

    # mem here is %; convert roughly using 40960 MiB if we only have %
    def pct_to_mb(pct):
        return pct * 40960 / 100.0 if pct is not None else None

    return (
        avg(utils[0]),
        avg(utils[1]),
        pct_to_mb(peak(mems[0])),
        pct_to_mb(peak(mems[1])),
    )


def parse_training_log(path: Path, warmup: int = 20) -> dict:
    step_ms = []
    peak_alloc = []
    for line in path.read_text(errors="ignore").splitlines():
        m = ITER_RE.search(line)
        if m:
            it = int(m.group(1))
            ms = float(m.group(2))
            if it > warmup:
                step_ms.append(ms)
        m2 = MEM_RE.search(line)
        if m2:
            peak_alloc.append(float(m2.group(1)))
    if not step_ms:
        return {}
    step_ms_sorted = sorted(step_ms)
    p95 = step_ms_sorted[min(len(step_ms_sorted) - 1, int(math.ceil(0.95 * len(step_ms_sorted)) - 1))]
    return {
        "n_measured": len(step_ms),
        "mean_step_ms": statistics.mean(step_ms),
        "median_step_ms": statistics.median(step_ms),
        "p95_step_ms": p95,
        "peak_alloc_gb": max(peak_alloc) if peak_alloc else None,
    }


CONFIGS = [
    # run_id, config, num_gpus, tp, pp, dp, micro, global, seq
    ("baseline_1gpu", "1GPU", 1, 1, 1, 1, 1, 64, 2048),
    ("dp2_strong", "DP2", 2, 1, 1, 2, 1, 64, 2048),
    ("dp2_weak", "DP2_weak", 2, 1, 1, 2, 1, 128, 2048),
    ("tp2", "TP2", 2, 2, 1, 1, 1, 64, 2048),
    ("pp2_mbs1", "PP2_mbs1", 2, 1, 2, 1, 1, 64, 2048),
    ("pp2_mbs2", "PP2_mbs2", 2, 1, 2, 1, 2, 64, 2048),
    ("pp2_mbs4", "PP2_mbs4", 2, 1, 2, 1, 4, 64, 2048),
]


def main() -> None:
    rows = []
    baseline_tps = None
    for run_id, config, ng, tp, pp, dp, mbs, gbs, seq in CONFIGS:
        log = LOG_DIR / f"{run_id}.log"
        if not log.exists():
            continue
        metrics = parse_training_log(log, warmup=20)
        if not metrics:
            print(f"WARN: no iteration metrics in {log}")
            continue
        u0, u1, m0, m1 = parse_dmon(LOG_DIR / f"{run_id}.dmon.csv")
        tokens_per_step = gbs * seq
        mean_s = metrics["mean_step_ms"] / 1000.0
        tps = tokens_per_step / mean_s
        tps_gpu = tps / ng
        if config == "1GPU":
            baseline_tps = tps
        rows.append(
            {
                "run_id": run_id,
                "config": config,
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
                "tokens_per_sec_per_gpu": round(tps_gpu, 1),
                "gpu0_peak_mem_mb": round(m0, 1) if m0 is not None else "",
                "gpu1_peak_mem_mb": round(m1, 1) if m1 is not None else "",
                "gpu0_avg_util": round(u0, 1) if u0 is not None else "",
                "gpu1_avg_util": round(u1, 1) if u1 is not None else "",
                "log_peak_alloc_gb": metrics.get("peak_alloc_gb") or "",
            }
        )

    out = RESULT_DIR / "training_results.csv"
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out}")

    print("\nComparison (vs 1GPU baseline when available):")
    print(
        f"{'Config':12} {'GPUs':>4} {'TP':>3} {'PP':>3} {'DP':>3} "
        f"{'Tok/s':>10} {'Speedup':>8} {'Tok/s/GPU':>10} {'step_ms':>8}"
    )
    for r in rows:
        if r["config"] == "DP2_weak":
            continue
        sp = (r["tokens_per_sec"] / baseline_tps) if baseline_tps else float("nan")
        print(
            f"{r['config']:12} {r['num_gpus']:4d} {r['tp']:3d} {r['pp']:3d} {r['dp']:3d} "
            f"{r['tokens_per_sec']:10.1f} {sp:7.2f}x {r['tokens_per_sec_per_gpu']:10.1f} "
            f"{r['mean_step_ms']:8.1f}"
        )


if __name__ == "__main__":
    main()
