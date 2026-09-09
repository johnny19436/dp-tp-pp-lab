#!/usr/bin/env python3
"""Generate figures + report_4gpu.pdf for the 4-GPU Megatron parallelism lab."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

LAB = Path(__file__).resolve().parents[2]
FIG = LAB / "analysis" / "4gpu" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = LAB / "results" / "4gpu"
OUT_PDF = LAB / "reports" / "4gpu" / "report_4gpu.pdf"
OUT_PDF_ROOT = LAB / "report_4gpu.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)


def load():
    train = pd.read_csv(RES / "training_results_4gpu.csv")
    nccl = pd.read_csv(RES / "nccl_results_4gpu.csv")
    cap = pd.read_csv(RES / "capacity_results_4gpu.csv") if (RES / "capacity_results_4gpu.csv").exists() else pd.DataFrame()
    nsys = pd.read_csv(RES / "nsys_comm_summary_4gpu.csv") if (RES / "nsys_comm_summary_4gpu.csv").exists() else pd.DataFrame()
    old_nccl = pd.read_csv(LAB / "results" / "nccl_results.csv") if (LAB / "results" / "nccl_results.csv").exists() else None
    return train, nccl, cap, nsys, old_nccl


def style_ax(ax, title: str):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def get(train, config):
    hit = train[train["config"] == config]
    if hit.empty:
        return None
    return hit.iloc[0]


def plot_dp_strong(train):
    configs = ["orig_1GPU", "orig_DP2", "orig_DP4"]
    rows = [get(train, c) for c in configs]
    rows = [r for r in rows if r is not None]
    if len(rows) < 2:
        return None
    labels = [r["config"].replace("orig_", "") for r in rows]
    tps = [r["tokens_per_sec"] for r in rows]
    base = tps[0]
    speedup = [t / base for t in tps]
    eff = [s / r["num_gpus"] for s, r in zip(speedup, rows)]
    colors_bar = ["#4C78A8", "#F58518", "#E45756"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    axes[0].bar(labels, np.array(tps) / 1000, color=colors_bar[: len(labels)], width=0.7)
    axes[0].set_ylabel("Throughput (k tokens/s)")
    style_ax(axes[0], "Original model DP throughput")

    axes[1].bar(labels, speedup, color=colors_bar[: len(labels)], width=0.7)
    axes[1].axhline(1, color="gray", ls=":")
    axes[1].axhline(2, color="gray", ls=":", alpha=0.4)
    axes[1].axhline(4, color="gray", ls=":", alpha=0.4)
    axes[1].set_ylabel("Speedup vs 1GPU")
    style_ax(axes[1], "Strong-scaling speedup")
    for i, v in enumerate(speedup):
        axes[1].text(i, v + 0.05, f"{v:.2f}×", ha="center", fontsize=8)

    axes[2].bar(labels, [e * 100 for e in eff], color=colors_bar[: len(labels)], width=0.7)
    axes[2].axhline(100, color="gray", ls=":")
    axes[2].set_ylabel("Efficiency (%)")
    axes[2].set_ylim(0, 110)
    style_ax(axes[2], "Strong-scaling efficiency")
    for i, e in enumerate(eff):
        axes[2].text(i, e * 100 + 1.5, f"{e*100:.1f}%", ha="center", fontsize=8)

    fig.tight_layout()
    path = FIG / "fig1_dp_strong.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dp_weak(train):
    one = get(train, "orig_1GPU")
    w2 = get(train, "orig_DP2_weak")
    w4 = get(train, "orig_DP4_weak")
    if one is None or w2 is None:
        return None
    labels, steps, tps = ["1GPU\nGBS=64"], [one["mean_step_ms"]], [one["tokens_per_sec"]]
    if w2 is not None:
        labels.append("DP2 weak\nGBS=128")
        steps.append(w2["mean_step_ms"])
        tps.append(w2["tokens_per_sec"])
    if w4 is not None:
        labels.append("DP4 weak\nGBS=256")
        steps.append(w4["mean_step_ms"])
        tps.append(w4["tokens_per_sec"])
    cols = ["#4C78A8", "#72B7B2", "#54A24B"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(labels, steps, color=cols[: len(labels)], width=0.65)
    axes[0].set_ylabel("Mean step time (ms)")
    style_ax(axes[0], "DP weak-scaling step time")
    for i, v in enumerate(steps):
        axes[0].text(i, v + max(steps) * 0.01, f"{v:.0f}", ha="center", fontsize=8)

    axes[1].bar(labels, np.array(tps) / 1000, color=cols[: len(labels)], width=0.65)
    axes[1].set_ylabel("Throughput (k tokens/s)")
    style_ax(axes[1], "DP weak-scaling throughput")
    # efficiency: step_1 / step_N (ideal = 1)
    if w4 is not None:
        eff4 = one["mean_step_ms"] / w4["mean_step_ms"]
        axes[1].text(0.5, 0.02, f"Weak eff@4 ≈ {eff4*100:.1f}% (t1/t4)", transform=axes[1].transAxes,
                     ha="center", fontsize=8, color="#333")
    fig.tight_layout()
    path = FIG / "fig2_dp_weak.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_nccl(nccl, old_nccl):
    fig, axes = plt.subplots(1, 2 if old_nccl is not None else 1, figsize=(11, 3.8))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax = axes[0]
    for name, color in [("all_reduce", "#4C78A8"), ("all_gather", "#F58518"), ("reduce_scatter", "#54A24B")]:
        d = nccl[nccl["collective"] == name]
        ax.plot(d["size_bytes"] / (1024**2), d["busbw_GBs"], "o-", label=name.replace("_", " "), color=color, lw=2, ms=5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message size (MiB)")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "NCCL 4-GPU microbenchmark")

    if old_nccl is not None:
        ax = axes[1]
        for name, color, ls in [("all_reduce", "#4C78A8", "-"), ("all_gather", "#F58518", "-")]:
            d4 = nccl[nccl["collective"] == name]
            d2 = old_nccl[old_nccl["collective"] == name]
            ax.plot(d2["size_bytes"] / (1024**2), d2["busbw_GBs"], "s--", color=color, alpha=0.7, label=f"2GPU {name}")
            ax.plot(d4["size_bytes"] / (1024**2), d4["busbw_GBs"], "o-", color=color, label=f"4GPU {name}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Message size (MiB)")
        ax.set_ylabel("Bus bandwidth (GB/s)")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        style_ax(ax, "2GPU vs 4GPU NCCL (not 2×)")
    fig.tight_layout()
    path = FIG / "fig3_nccl.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_capacity(cap):
    if cap is None or cap.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.4))
    labels = cap["config"].tolist()
    mem = []
    cols = []
    for _, r in cap.iterrows():
        if r["status"] == "OOM":
            mem.append(40.0)
            cols.append("#E45756")
        else:
            mb = r["peak_alloc_mb"]
            mem.append(float(mb) / 1024 if mb == mb and mb != "" else 0)
            cols.append("#54A24B")
    ax.bar(labels, mem, color=cols, width=0.65)
    ax.axhline(40, color="#B279A2", ls="--", label="A100 40GB")
    ax.set_ylabel("Peak allocated (GB) / OOM→40")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "Large-model capacity (smoke)")
    fig.tight_layout()
    path = FIG / "fig4_capacity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_large_throughput(train):
    order = [
        "large_TP2",
        "large_TP2DP2_strong",
        "large_PP2",
        "large_PP2DP2_strong",
        "large_TP4",
        "large_PP4_mbs1",
        "large_TP2PP2",
    ]
    rows = [get(train, c) for c in order]
    present = [(c, r) for c, r in zip(order, rows) if r is not None]
    if not present:
        return None
    labels = [c.replace("large_", "").replace("_strong", "") for c, _ in present]
    tps = [r["tokens_per_sec"] for _, r in present]
    tpsg = [r["tokens_per_sec_per_gpu"] for _, r in present]
    mem = [float(r["peak_alloc_mb"]) / 1024 if r["peak_alloc_mb"] == r["peak_alloc_mb"] and r["peak_alloc_mb"] != "" else 0 for _, r in present]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(range(len(labels)), np.array(tps) / 1000, color="#4C78A8")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("k tokens/s")
    style_ax(axes[0], "Large-model throughput")

    axes[1].bar(range(len(labels)), np.array(tpsg) / 1000, color="#F58518")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("k tokens/s/GPU")
    style_ax(axes[1], "Throughput per GPU")

    axes[2].bar(range(len(labels)), mem, color="#54A24B")
    axes[2].axhline(40, color="#B279A2", ls="--")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("Peak alloc (GB)")
    style_ax(axes[2], "Peak VRAM / GPU")
    fig.tight_layout()
    path = FIG / "fig5_large_main.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pp_sweep(train):
    configs = ["large_PP2", "large_PP4_mbs1", "large_PP4_mbs2", "large_PP4_mbs4"]
    rows = [(c, get(train, c)) for c in configs]
    rows = [(c, r) for c, r in rows if r is not None]
    if len(rows) < 2:
        return None
    labels = [c.replace("large_", "") for c, _ in rows]
    tps = [r["tokens_per_sec"] for _, r in rows]
    mem = [float(r["peak_alloc_mb"]) / 1024 if r["peak_alloc_mb"] else 0 for _, r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(range(len(labels)), np.array(tps) / 1000, "o-", color="#54A24B", lw=2, ms=8)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("k tokens/s")
    style_ax(axes[0], "PP2 vs PP4 microbatch sweep")
    axes[1].plot(range(len(labels)), mem, "s-", color="#B279A2", lw=2, ms=8)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("Peak alloc (GB)")
    style_ax(axes[1], "PP VRAM vs config")
    fig.tight_layout()
    path = FIG / "fig6_pp_sweep.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_nsys(nsys):
    if nsys is None or nsys.empty:
        return None
    d = nsys[nsys["nccl_kernel_time_pct_est"] != ""].copy()
    if d.empty:
        return None
    d["nccl_kernel_time_pct_est"] = d["nccl_kernel_time_pct_est"].astype(float)
    labels = [x.replace("nsys_", "") for x in d["profile"]]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(range(len(labels)), d["nccl_kernel_time_pct_est"], color="#E45756", width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% CUDA kernel time (NCCL est.)")
    style_ax(ax, "Nsight: NCCL communication share")
    fig.tight_layout()
    path = FIG / "fig7_nsys.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tp_scale(train):
    tp2 = get(train, "large_TP2")
    tp4 = get(train, "large_TP4")
    if tp2 is None or tp4 is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    labels = ["TP2", "TP4"]
    axes[0].bar(labels, [tp2["tokens_per_sec"] / 1000, tp4["tokens_per_sec"] / 1000], color=["#E45756", "#B279A2"])
    axes[0].set_ylabel("k tokens/s")
    style_ax(axes[0], "TP2 vs TP4 throughput")
    axes[1].bar(labels, [float(tp2["peak_alloc_mb"]) / 1024, float(tp4["peak_alloc_mb"]) / 1024], color=["#E45756", "#B279A2"])
    axes[1].set_ylabel("Peak alloc (GB)")
    style_ax(axes[1], "TP2 vs TP4 memory")
    fig.tight_layout()
    path = FIG / "fig8_tp_scale.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fb")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def build_pdf(figs, train, nccl, cap, nsys):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"], fontSize=18, spaceAfter=6))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#444"), alignment=TA_CENTER, spaceAfter=14))
    styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], fontSize=13, textColor=colors.HexColor("#1f4e79"), spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], fontSize=11, textColor=colors.HexColor("#2e75b6"), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="BodyJ", parent=styles["BodyText"], fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#444"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="CodeSmall", parent=styles["Code"], fontSize=7.5, leading=9))

    story = []
    story.append(Paragraph("Single-Node 4×A100 Parallelism Experiment Report", styles["TitleBig"]))
    story.append(Paragraph("Scaling and Multi-Dimensional Parallelism with Megatron-LM", styles["Sub"]))

    # --- Executive summary ---
    one = get(train, "orig_1GPU")
    dp2 = get(train, "orig_DP2")
    dp4 = get(train, "orig_DP4")
    bullets = []
    if one is not None and dp4 is not None:
        sp4 = dp4["tokens_per_sec"] / one["tokens_per_sec"]
        ef4 = sp4 / 4
        bullets.append(f"Original-model DP4 strong scaling: {dp4['tokens_per_sec']:.0f} tok/s ({sp4:.2f}× vs 1GPU, efficiency {ef4*100:.1f}%).")
    w4 = get(train, "orig_DP4_weak")
    if one is not None and w4 is not None:
        we = one["mean_step_ms"] / w4["mean_step_ms"]
        bullets.append(f"DP4 weak-scaling efficiency (t1/t4): {we*100:.1f}% with GBS 64→256.")
    tp2 = get(train, "large_TP2")
    tp2dp2w = get(train, "large_TP2DP2_weak")
    pp2 = get(train, "large_PP2")
    pp2dp2w = get(train, "large_PP2DP2_weak")
    if not cap.empty:
        c1 = cap[cap["config"] == "cap_1GPU"]
        if not c1.empty:
            bullets.append(f"Large-model 1GPU capacity: {c1.iloc[0]['status']} (DP alone does not reduce replica memory).")
    large_rows = train[train["model_id"] == "large"]
    if not large_rows.empty:
        best = large_rows.loc[large_rows["tokens_per_sec"].idxmax()]
        bullets.append(f"Highest large-model aggregate throughput: {best['config']} at {best['tokens_per_sec']:.0f} tok/s.")
    story.append(Paragraph("1. Executive Summary", styles["H1c"]))
    story.append(Paragraph(
        "This report extends the prior 2×A100 DP/TP/PP lab to a single 4×A100-SXM4-40GB node. "
        "Question A reuses the original ~1.3B GPT config to measure pure DP scaling 1→2→4. "
        "Question B introduces a larger model that requires model parallelism, then compares TP, PP, "
        "and 2D combinations (TP×DP, PP×DP, TP×PP, TP4, PP4).",
        styles["BodyJ"],
    ))
    if bullets:
        story.append(ListFlowable([ListItem(Paragraph(b, styles["BodyJ"])) for b in bullets], bulletType="bullet", leftIndent=12))

    # --- Hardware ---
    story.append(Paragraph("2. Hardware and Software Setup", styles["H1c"]))
    topo = (LAB / "logs" / "4gpu" / "00_system_check.log").read_text(errors="ignore") if (LAB / "logs" / "4gpu" / "00_system_check.log").exists() else ""
    story.append(Paragraph(
        "Four NVIDIA A100-SXM4-40GB GPUs on one node. Topology matrix shows <b>NV12</b> between every GPU pair "
        "(full NVLink clique). GPUs 0–1 share NUMA 3; GPUs 2–3 share NUMA 1 — still NVLink-connected, so "
        "collectives stay on NVLink rather than traversing SYS for GPU↔GPU traffic.",
        styles["BodyJ"],
    ))
    story.append(Paragraph(
        "Software: PyTorch 2.10 (NGC), Megatron-LM <font face='Courier'>be85fc5df</font>, BF16, "
        "<font face='Courier'>--mock-data</font>, FlashAttention enabled. Exact dumps: <font face='Courier'>logs/4gpu/00_system_check.log</font>.",
        styles["BodyJ"],
    ))

    story.append(Paragraph("3. What Changed from the 2-GPU Experiment", styles["H1c"]))
    story.append(Paragraph(
        "The 2-GPU artifacts (<font face='Courier'>report.pdf</font>, CSVs, logs, profiles) are preserved untouched. "
        "This generation writes only under <font face='Courier'>logs/4gpu/</font>, <font face='Courier'>results/4gpu/</font>, "
        "<font face='Courier'>profiles/4gpu/</font>, and <font face='Courier'>reports/4gpu/</font>. "
        "New work: DP4 scaling on the original model; a larger model for capacity/model-parallel study; "
        "multi-dimensional layouts TP2×DP2, PP2×DP2, TP4, PP4, TP2×PP2; expanded Nsight set.",
        styles["BodyJ"],
    ))

    # --- NCCL ---
    story.append(Paragraph("4. 4-GPU NCCL Baseline", styles["H1c"]))
    story.append(Paragraph(
        "AllReduce / AllGather / ReduceScatter from 8 MiB→1 GiB on 4 GPUs. Bus bandwidth must not be read as "
        "“twice the 2-GPU number”: increasing N changes both communication volume and collective scheduling.",
        styles["BodyJ"],
    ))
    if figs.get("nccl"):
        story.append(Image(str(figs["nccl"]), width=6.8 * inch, height=2.3 * inch))
        story.append(Paragraph("Figure: 4-GPU NCCL busbw and 2GPU vs 4GPU comparison.", styles["Caption"]))
    # peak 1G row
    if not nccl.empty:
        ar = nccl[(nccl["collective"] == "all_reduce") & (nccl["size_bytes"] == 1073741824)]
        if not ar.empty:
            story.append(Paragraph(
                f"At 1 GiB AllReduce: time={ar.iloc[0]['time_us']:.1f} µs, algbw={ar.iloc[0]['algbw_GBs']:.1f} GB/s, "
                f"busbw={ar.iloc[0]['busbw_GBs']:.1f} GB/s (CSV: results/4gpu/nccl_results_4gpu.csv).",
                styles["BodyJ"],
            ))

    # --- Original DP ---
    story.append(Paragraph("5. Original Model: DP Scaling 1 → 2 → 4", styles["H1c"]))
    story.append(Paragraph(
        "Frozen original config: layers=24, hidden=2048, ffn=8192, heads=16, seq=2048, vocab=32000, BF16, "
        "micro_batch=1, global_batch=64 (strong). Warmup=20, measured=100.",
        styles["BodyJ"],
    ))
    if figs.get("dp_strong"):
        story.append(Image(str(figs["dp_strong"]), width=6.8 * inch, height=2.15 * inch))
        story.append(Paragraph("Figure: DP strong scaling throughput, speedup, efficiency.", styles["Caption"]))
    # table
    hdr = ["Config", "GPUs", "Tok/s", "Speedup", "Eff%", "step_ms", "Tok/s/GPU", "peak_MB"]
    data = [hdr]
    base_tps = one["tokens_per_sec"] if one is not None else None
    for c in ["orig_1GPU", "orig_DP2", "orig_DP4"]:
        r = get(train, c)
        if r is None:
            continue
        sp = r["tokens_per_sec"] / base_tps if base_tps else float("nan")
        ef = 100 * sp / r["num_gpus"] if base_tps else float("nan")
        data.append([c, int(r["num_gpus"]), fmt(r["tokens_per_sec"], 0), fmt(sp, 2), fmt(ef, 1),
                     fmt(r["mean_step_ms"], 1), fmt(r["tokens_per_sec_per_gpu"], 0), fmt(r["peak_alloc_mb"], 0)])
    story.append(table(data))
    story.append(Spacer(1, 8))
    story.append(Paragraph("5.1 Weak scaling", styles["H2c"]))
    story.append(Paragraph(
        "Weak scaling keeps work/GPU ≈ constant: GBS=64,128,256 for DP=1,2,4. "
        "Efficiency defined as <b>t_1GPU / t_DP</b> (ideal ≈ 1.0 when step time stays flat).",
        styles["BodyJ"],
    ))
    if figs.get("dp_weak"):
        story.append(Image(str(figs["dp_weak"]), width=6.5 * inch, height=2.3 * inch))
        story.append(Paragraph("Figure: DP weak-scaling step time and throughput.", styles["Caption"]))

    # --- Large model ---
    story.append(PageBreak())
    story.append(Paragraph("6. Large Model Configuration", styles["H1c"]))
    cfg = (RES / "large_model_config.txt").read_text(errors="ignore") if (RES / "large_model_config.txt").exists() else "(see results/4gpu/large_model_config.txt)"
    story.append(Preformatted(cfg[:1200], styles["CodeSmall"]))
    story.append(Paragraph(
        "Design goal: 1GPU (and therefore pure DP) should OOM or sit beyond a safe margin; TP2/PP2 must fit. "
        "Any size adjustment is documented in the config file and frozen before comparable large-model runs.",
        styles["BodyJ"],
    ))

    story.append(Paragraph("7. Capacity Results", styles["H1c"]))
    if figs.get("capacity"):
        story.append(Image(str(figs["capacity"]), width=5.8 * inch, height=2.4 * inch))
        story.append(Paragraph("Figure: capacity smoke outcomes (red≈OOM).", styles["Caption"]))
    if not cap.empty:
        cdata = [["Config", "GPUs", "TP", "PP", "DP", "Status", "peak_MB"]]
        for _, r in cap.iterrows():
            cdata.append([r["config"], int(r["num_gpus"]), int(r["tp"]), int(r["pp"]), int(r["dp"]),
                          r["status"], fmt(r["peak_alloc_mb"], 0) if r["peak_alloc_mb"] != "" else "—"])
        story.append(table(cdata))
        story.append(Paragraph(
            "Pure DP4 does not fix single-replica OOM: each rank still holds a full model copy. "
            "TP shards weights/activations within a layer; PP shards layers across stages.",
            styles["BodyJ"],
        ))

    story.append(Paragraph("8. TP2 and TP2×DP2", styles["H1c"]))
    story.append(Paragraph(
        "TP2 baseline on 2 GPUs; TP2×DP2 places two TP replicas on 4 GPUs. Strong: same GBS as TP2. "
        "Weak: 2× GBS. Weak efficiency = throughput_TP2DP2 / (2 × throughput_TP2).",
        styles["BodyJ"],
    ))
    for c in ["large_TP2", "large_TP2DP2_strong", "large_TP2DP2_weak"]:
        r = get(train, c)
        if r is not None:
            story.append(Paragraph(
                f"<b>{c}</b>: {r['tokens_per_sec']:.0f} tok/s, step={r['mean_step_ms']:.1f} ms, "
                f"peak_alloc={r['peak_alloc_mb']} MB, tok/s/GPU={r['tokens_per_sec_per_gpu']:.0f}.",
                styles["BodyJ"],
            ))
    if tp2 is not None and tp2dp2w is not None:
        we = tp2dp2w["tokens_per_sec"] / (2 * tp2["tokens_per_sec"])
        story.append(Paragraph(f"TP2×DP2 weak-scaling efficiency: {we*100:.1f}%.", styles["BodyJ"]))

    story.append(Paragraph("9. PP2 and PP2×DP2", styles["H1c"]))
    for c in ["large_PP2", "large_PP2DP2_strong", "large_PP2DP2_weak"]:
        r = get(train, c)
        if r is not None:
            story.append(Paragraph(
                f"<b>{c}</b>: {r['tokens_per_sec']:.0f} tok/s, step={r['mean_step_ms']:.1f} ms, "
                f"peak_alloc={r['peak_alloc_mb']} MB.",
                styles["BodyJ"],
            ))
    if pp2 is not None and pp2dp2w is not None:
        we = pp2dp2w["tokens_per_sec"] / (2 * pp2["tokens_per_sec"])
        story.append(Paragraph(f"PP2×DP2 weak-scaling efficiency: {we*100:.1f}%.", styles["BodyJ"]))
    if tp2dp2w is not None and pp2dp2w is not None:
        winner = "TP2×DP2" if tp2dp2w["tokens_per_sec"] >= pp2dp2w["tokens_per_sec"] else "PP2×DP2"
        story.append(Paragraph(
            f"Head-to-head weak aggregate: {winner} wins "
            f"({max(tp2dp2w['tokens_per_sec'], pp2dp2w['tokens_per_sec']):.0f} vs "
            f"{min(tp2dp2w['tokens_per_sec'], pp2dp2w['tokens_per_sec']):.0f} tok/s).",
            styles["BodyJ"],
        ))

    story.append(Paragraph("10. TP2 vs TP4", styles["H1c"]))
    if figs.get("tp_scale"):
        story.append(Image(str(figs["tp_scale"]), width=5.8 * inch, height=2.2 * inch))
        story.append(Paragraph("Figure: TP degree scaling.", styles["Caption"]))
    tp4 = get(train, "large_TP4")
    if tp2 is not None and tp4 is not None:
        story.append(Paragraph(
            f"TP4 vs TP2 throughput ratio={tp4['tokens_per_sec']/tp2['tokens_per_sec']:.2f}; "
            f"memory TP2={float(tp2['peak_alloc_mb'])/1024:.1f} GB → TP4={float(tp4['peak_alloc_mb'])/1024:.1f} GB. "
            "Higher TP usually raises NCCL share and lowers tok/s/GPU.",
            styles["BodyJ"],
        ))

    story.append(Paragraph("11. PP2 vs PP4 (microbatch sweep)", styles["H1c"]))
    if figs.get("pp_sweep"):
        story.append(Image(str(figs["pp_sweep"]), width=6.3 * inch, height=2.2 * inch))
        story.append(Paragraph("Figure: pipeline depth and microbatch vs throughput/VRAM.", styles["Caption"]))
    story.append(Paragraph(
        "Deeper PP increases bubble fraction for a fixed number of microbatches; raising micro_batch_size "
        "fills the pipeline and can improve GEMM efficiency, at the cost of activation memory. "
        "Fair comparisons keep micro_batch=1; sweeps are sensitivity, not pure scaling.",
        styles["BodyJ"],
    ))

    story.append(Paragraph("12. TP2×PP2", styles["H1c"]))
    hyb = get(train, "large_TP2PP2")
    if hyb is not None:
        story.append(Paragraph(
            f"TP2×PP2 on 4 GPUs: {hyb['tokens_per_sec']:.0f} tok/s, step={hyb['mean_step_ms']:.1f} ms, "
            f"peak_alloc={hyb['peak_alloc_mb']} MB. Combines intra-layer TP collectives with inter-stage P2P.",
            styles["BodyJ"],
        ))

    if figs.get("large_main"):
        story.append(Paragraph("Large-model overview", styles["H2c"]))
        story.append(Image(str(figs["large_main"]), width=6.8 * inch, height=2.2 * inch))
        story.append(Paragraph("Figure: large-model throughput, per-GPU throughput, VRAM.", styles["Caption"]))

    story.append(PageBreak())
    story.append(Paragraph("13. Nsight Systems Communication Analysis", styles["H1c"]))
    if figs.get("nsys"):
        story.append(Image(str(figs["nsys"]), width=6.5 * inch, height=2.2 * inch))
        story.append(Paragraph("Figure: estimated NCCL share of CUDA kernel time.", styles["Caption"]))
    story.append(Paragraph(
        "Profiles are short steady-state windows (TRAIN_ITERS=30, nsys duration≈50s). "
        "DP is dominated by infrequent gradient AllReduce; TP shows frequent bf16 AllReduce/AllGather/"
        "ReduceScatter inside layers; PP emphasizes SendRecv with compute bubbles; hybrids mix both patterns. "
        "Raw: profiles/4gpu/*.nsys-rep and *_stats.txt.",
        styles["BodyJ"],
    ))

    story.append(Paragraph("14. Overall Throughput / Memory / Communication Tradeoff", styles["H1c"]))
    story.append(Paragraph(
        "On this NVLink clique: prefer pure DP while the replica fits; introduce TP or PP only for capacity; "
        "use remaining GPUs for DP (TP×DP / PP×DP) rather than maximizing TP degree alone; treat PP microbatch "
        "as a tuning knob separate from fair scaling claims.",
        styles["BodyJ"],
    ))

    story.append(Paragraph("15. Answers to Study Questions", styles["H1c"]))
    answers = build_answers(train, cap, nsys, one, dp2, dp4, w4, tp2, tp2dp2w, pp2, pp2dp2w, tp4, hyb)
    for q, a in answers:
        story.append(Paragraph(f"<b>{q}</b> {a}", styles["BodyJ"]))

    story.append(Paragraph("16. Conclusions and Practical Guidance", styles["H1c"]))
    story.append(Paragraph(
        "DP scales nearly linearly on this 4-GPU NV12 node for the original model. For models that exceed 40 GB "
        "per replica, shard with TP or PP first, then scale out with DP. Avoid TP4 unless memory demands it—"
        "communication share rises quickly. Prefer PP microbatch sweeps after choosing the parallel layout.",
        styles["BodyJ"],
    ))

    story.append(Paragraph("17. Reproducibility", styles["H1c"]))
    story.append(Preformatted(
        "bash scripts/setup_third_party.sh\n"
        "bash run_all_4gpu.sh\n"
        "# or stepwise scripts under scripts/4gpu/\n"
        "python3 analysis/4gpu/summarize_4gpu.py\n"
        "python3 analysis/4gpu/generate_report_4gpu.py\n"
        "# outputs: results/4gpu/*.csv, reports/4gpu/report_4gpu.pdf, report_4gpu.pdf",
        styles["CodeSmall"],
    ))
    story.append(Paragraph(
        "All numeric claims trace to logs/4gpu/*.log, results/4gpu/*.csv, and profiles/4gpu/*_stats.txt. "
        "Legacy 2-GPU report.pdf is unchanged.",
        styles["BodyJ"],
    ))

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.65 * inch, bottomMargin=0.65 * inch)
    doc.build(story)
    # also copy to repo root for convenience
    OUT_PDF_ROOT.write_bytes(OUT_PDF.read_bytes())
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PDF_ROOT}")


def build_answers(train, cap, nsys, one, dp2, dp4, w4, tp2, tp2dp2w, pp2, pp2dp2w, tp4, hyb):
    def s(r, key="tokens_per_sec"):
        return "n/a" if r is None else f"{r[key]:.0f}"

    ans = []
    if one is not None and dp4 is not None:
        sp = dp4["tokens_per_sec"] / one["tokens_per_sec"]
        ans.append(("1–2. DP 1→2→4 / DP4 efficiency:",
                    f"1GPU={s(one)}, DP2={s(dp2)}, DP4={s(dp4)} tok/s; speedup_4={sp:.2f}×, efficiency={100*sp/4:.1f}%."))
    else:
        ans.append(("1–2. DP scaling:", "see training_results_4gpu.csv"))
    if one is not None and w4 is not None:
        ans.append(("3. DP4 weak efficiency:", f"t1/t4 = {one['mean_step_ms']/w4['mean_step_ms']*100:.1f}%."))
    if not cap.empty:
        def st(name):
            h = cap[cap["config"] == name]
            return h.iloc[0]["status"] if not h.empty else "?"
        ans.append(("4–5. Large 1GPU / DP4 fit?:", f"1GPU={st('cap_1GPU')}, DP4={st('cap_DP4')} — DP does not shrink replica memory."))
        ans.append(("6–7. TP2/PP2 memory:", f"TP2={st('cap_TP2')}, PP2={st('cap_PP2')} (peak MB in capacity CSV)."))
    if tp2 is not None and tp2dp2w is not None:
        ans.append(("8. TP2×DP2 vs TP2:", f"weak eff={tp2dp2w['tokens_per_sec']/(2*tp2['tokens_per_sec'])*100:.1f}%."))
    if pp2 is not None and pp2dp2w is not None:
        ans.append(("9. PP2×DP2 vs PP2:", f"weak eff={pp2dp2w['tokens_per_sec']/(2*pp2['tokens_per_sec'])*100:.1f}%."))
    if tp2dp2w is not None and pp2dp2w is not None:
        ans.append(("10. TP2×DP2 vs PP2×DP2:", f"{'TP2×DP2' if tp2dp2w['tokens_per_sec']>=pp2dp2w['tokens_per_sec'] else 'PP2×DP2'} higher aggregate weak throughput."))
    if tp2 is not None and tp4 is not None:
        ans.append(("11–12. TP2→TP4:", f"throughput ratio={tp4['tokens_per_sec']/tp2['tokens_per_sec']:.2f}; see Nsight CSV for NCCL share."))
    ans.append(("13–14. PP bubbles / microbatch:", "PP4 needs more microbatches to hide bubbles; sweep shows throughput/VRAM tradeoff (fig6)."))
    ans.append(("15. TP2×PP2 Nsight:", "mixed AllReduce + SendRecv; see nsys_large_tp2pp2_stats.txt."))
    large = train[train["model_id"] == "large"] if "model_id" in train.columns else train
    if not large.empty:
        best = large.loc[large["tokens_per_sec"].idxmax()]
        bestg = large.loc[large["tokens_per_sec_per_gpu"].idxmax()]
        # lowest memory among successful
        mem = large[large["peak_alloc_mb"].astype(str) != ""]
        if not mem.empty:
            low = mem.loc[mem["peak_alloc_mb"].astype(float).idxmin()]
            ans.append(("16–18. Best tok/s / mem / tok/s/GPU:",
                        f"max tok/s={best['config']} ({best['tokens_per_sec']:.0f}); "
                        f"min mem={low['config']} ({low['peak_alloc_mb']} MB); "
                        f"max tok/s/GPU={bestg['config']} ({bestg['tokens_per_sec_per_gpu']:.0f})."))
    ans.append(("19. Most frequent communication:", "Pure TP (and TP4) — per-layer collectives; see Nsight NCCL %."))
    ans.append(("20. Practical rule:",
                "Fit with DP if possible; else minimal TP/PP for capacity, then DP on remaining GPUs; "
                "avoid over-sharding TP; tune PP microbatch separately."))
    return ans


def main():
    train, nccl, cap, nsys, old_nccl = load()
    figs = {
        "dp_strong": plot_dp_strong(train),
        "dp_weak": plot_dp_weak(train),
        "nccl": plot_nccl(nccl, old_nccl),
        "capacity": plot_capacity(cap),
        "large_main": plot_large_throughput(train),
        "pp_sweep": plot_pp_sweep(train),
        "nsys": plot_nsys(nsys),
        "tp_scale": plot_tp_scale(train),
    }
    build_pdf(figs, train, nccl, cap, nsys)


if __name__ == "__main__":
    main()
