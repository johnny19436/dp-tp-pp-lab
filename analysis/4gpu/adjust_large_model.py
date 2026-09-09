#!/usr/bin/env python3
"""Inspect capacity logs; enlarge model if 1GPU unexpectedly fits with headroom."""
from __future__ import annotations

import re
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
LOG = LAB / "logs" / "4gpu"
RES = LAB / "results" / "4gpu"
ENV = RES / "large_model_config.env"
FLAG = RES / "NEED_RECAPACITY"

OOM_RE = re.compile(r"out of memory|CUDA out of memory|torch.OutOfMemoryError", re.IGNORECASE)
MEM_RE = re.compile(r"max allocated:\s*([\d.]+)", re.IGNORECASE)


def log_status(name: str):
    p = LOG / f"{name}.log"
    if not p.exists():
        return "missing", None
    text = p.read_text(errors="ignore")
    oom = bool(OOM_RE.search(text))
    mems = [float(x) for x in MEM_RE.findall(text)]
    peak = max(mems) if mems else None
    return ("OOM" if oom else "OK"), peak


def rewrite_env(**kwargs):
    lines = []
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if not line.startswith("export "):
                lines.append(line)
                continue
            key = line.split("=", 1)[0].replace("export ", "").strip()
            if key in kwargs:
                lines.append(f"export {key}={kwargs[key]}")
                kwargs.pop(key)
            else:
                lines.append(line)
    for k, v in kwargs.items():
        lines.append(f"export {k}={v}")
    ENV.write_text("\n".join(lines) + "\n")


def main():
    s1, m1 = log_status("cap_1gpu")
    stp, mtp = log_status("cap_tp2")
    spp, mpp = log_status("cap_pp2")
    print(f"cap_1gpu={s1} peak={m1}  cap_tp2={stp} peak={mtp}  cap_pp2={spp} peak={mpp}")

    FLAG.unlink(missing_ok=True)

    # If 1GPU fits with comfortable headroom (<36GB allocated), grow layers
    if s1 == "OK" and m1 is not None and m1 < 36000:
        print("1GPU fits with headroom — increasing LARGE_NUM_LAYERS by +4")
        # read current
        cur = 20
        if ENV.exists():
            for line in ENV.read_text().splitlines():
                if "LARGE_NUM_LAYERS=" in line:
                    cur = int(line.split("=")[1])
        rewrite_env(LARGE_NUM_LAYERS=cur + 4)
        FLAG.write_text("1\n")
        return

    # If TP2/PP2 OOM, shrink layers minimally
    if stp == "OOM" or spp == "OOM":
        print("TP2 or PP2 OOM — decreasing LARGE_NUM_LAYERS by -2")
        cur = 20
        if ENV.exists():
            for line in ENV.read_text().splitlines():
                if "LARGE_NUM_LAYERS=" in line:
                    cur = int(line.split("=")[1])
        rewrite_env(LARGE_NUM_LAYERS=max(12, cur - 2))
        FLAG.write_text("1\n")
        return

    print("Capacity looks consistent with plan goals; no size adjust.")


if __name__ == "__main__":
    main()
