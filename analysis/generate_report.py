#!/usr/bin/env python3
"""Generate plots + detailed report.pdf for the DP/TP/PP lab."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
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

LAB = Path("/root/working/dp-tp-pp-lab")
FIG = LAB / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT_PDF = LAB / "report.pdf"

# Canonical peak allocated MB from Megatron logs (more accurate than dmon %)
PEAK_ALLOC = {
    "1GPU": 34681.62,
    "DP2": 34681.60,
    "DP2_weak": 34681.62,
    "TP2": 17884.14,
    "PP2_mbs1": 19267.52,
    "PP2_mbs2": 23763.33,
    "PP2_mbs4": 32370.95,
}

NCCL_KERNEL_PCT = {"DP2": 1.3, "TP2": 37.2, "PP2": 7.0}  # dominant collective share


def load_data():
    train = pd.read_csv(LAB / "results" / "training_results.csv")
    nccl = pd.read_csv(LAB / "results" / "nccl_results.csv")
    baseline = float(train.loc[train["config"] == "1GPU", "tokens_per_sec"].iloc[0])
    train = train.copy()
    train["speedup"] = train["tokens_per_sec"] / baseline
    train["peak_alloc_mb"] = train["config"].map(PEAK_ALLOC)
    train["efficiency"] = np.where(
        train["num_gpus"] > 1,
        train["speedup"] / train["num_gpus"],
        1.0,
    )
    return train, nccl, baseline


def style_ax(ax, title: str):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_main_comparison(train: pd.DataFrame):
    main = train[train["config"].isin(["1GPU", "DP2", "TP2", "PP2_mbs1"])].copy()
    order = ["1GPU", "DP2", "TP2", "PP2_mbs1"]
    main["config"] = pd.Categorical(main["config"], order, ordered=True)
    main = main.sort_values("config")
    labels = ["1GPU", "DP2", "TP2", "PP2\n(mbs=1)"]
    colors_bar = ["#4C78A8", "#F58518", "#E45756", "#54A24B"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    ax = axes[0]
    ax.bar(labels, main["tokens_per_sec"] / 1000, color=colors_bar, width=0.7)
    ax.set_ylabel("Throughput (k tokens/s)")
    style_ax(ax, "Training throughput")

    ax = axes[1]
    ax.bar(labels, main["speedup"], color=colors_bar, width=0.7)
    ax.axhline(1.0, color="gray", lw=1, ls=":")
    ax.axhline(2.0, color="gray", lw=1, ls=":", alpha=0.5)
    ax.set_ylabel("Speedup vs 1GPU")
    style_ax(ax, "Strong-scaling speedup")
    for i, v in enumerate(main["speedup"]):
        ax.text(i, v + 0.04, f"{v:.2f}×", ha="center", fontsize=8)

    ax = axes[2]
    ax.bar(labels, main["peak_alloc_mb"] / 1024, color=colors_bar, width=0.7)
    ax.set_ylabel("Peak allocated (GB)")
    ax.axhline(40, color="#B279A2", lw=1.2, ls="--", label="A100 40GB")
    ax.legend(fontsize=8, frameon=False)
    style_ax(ax, "Per-GPU peak VRAM")

    fig.tight_layout()
    path = FIG / "fig1_main_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pp_sweep(train: pd.DataFrame):
    pp = train[train["config"].str.startswith("PP2")].copy()
    pp["mbs"] = pp["micro_batch"].astype(int)
    pp = pp.sort_values("mbs")
    dp = float(train.loc[train["config"] == "DP2", "tokens_per_sec"].iloc[0])
    base = float(train.loc[train["config"] == "1GPU", "tokens_per_sec"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(pp["mbs"], pp["tokens_per_sec"] / 1000, "o-", color="#54A24B", lw=2, ms=8)
    ax.axhline(dp / 1000, color="#F58518", ls="--", label=f"DP2 ({dp/1000:.1f}k)")
    ax.axhline(base / 1000, color="#4C78A8", ls=":", label=f"1GPU ({base/1000:.1f}k)")
    ax.set_xlabel("micro_batch_size")
    ax.set_ylabel("Throughput (k tokens/s)")
    ax.set_xticks([1, 2, 4])
    ax.legend(fontsize=8, frameon=False)
    style_ax(ax, "PP=2 throughput vs microbatch")

    ax = axes[1]
    ax.plot(pp["mbs"], pp["peak_alloc_mb"] / 1024, "s-", color="#B279A2", lw=2, ms=8)
    ax.set_xlabel("micro_batch_size")
    ax.set_ylabel("Peak allocated (GB)")
    ax.set_xticks([1, 2, 4])
    style_ax(ax, "PP=2 VRAM vs microbatch")
    for x, y in zip(pp["mbs"], pp["peak_alloc_mb"] / 1024):
        ax.text(x, y + 0.4, f"{y:.1f}G", ha="center", fontsize=8)

    fig.tight_layout()
    path = FIG / "fig2_pp_sweep.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_weak_scaling(train: pd.DataFrame):
    one = train.loc[train["config"] == "1GPU"].iloc[0]
    weak = train.loc[train["config"] == "DP2_weak"].iloc[0]
    strong = train.loc[train["config"] == "DP2"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ax = axes[0]
    labels = ["1GPU\nGBS=64", "DP2 strong\nGBS=64", "DP2 weak\nGBS=128"]
    vals = [one["mean_step_ms"], strong["mean_step_ms"], weak["mean_step_ms"]]
    cols = ["#4C78A8", "#F58518", "#72B7B2"]
    ax.bar(labels, vals, color=cols, width=0.65)
    ax.set_ylabel("Mean step time (ms)")
    style_ax(ax, "Step time: strong vs weak DP")
    for i, v in enumerate(vals):
        ax.text(i, v + 80, f"{v:.0f} ms", ha="center", fontsize=8)

    ax = axes[1]
    labels2 = ["1GPU", "DP2 strong", "DP2 weak"]
    tps = [one["tokens_per_sec"], strong["tokens_per_sec"], weak["tokens_per_sec"]]
    ax.bar(labels2, np.array(tps) / 1000, color=cols, width=0.65)
    ax.set_ylabel("Throughput (k tokens/s)")
    style_ax(ax, "Aggregate throughput")
    eff = one["mean_step_ms"] / weak["mean_step_ms"]
    ax.text(
        0.5,
        0.02,
        f"Weak-scaling efficiency ≈ {eff*100:.1f}%\n(step time nearly flat at 2× work)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    fig.tight_layout()
    path = FIG / "fig3_weak_scaling.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_nccl(nccl: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for name, color in [
        ("all_reduce", "#4C78A8"),
        ("all_gather", "#F58518"),
        ("reduce_scatter", "#54A24B"),
    ]:
        d = nccl[nccl["collective"] == name]
        ax.plot(
            d["size_bytes"] / (1024**2),
            d["busbw_GBs"],
            "o-",
            label=name.replace("_", " "),
            color=color,
            lw=2,
            ms=5,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message size (MiB)")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax, "NCCL 2-GPU NV12 microbenchmark")
    fig.tight_layout()
    path = FIG / "fig4_nccl.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comm_and_util(train: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    ax = axes[0]
    labels = list(NCCL_KERNEL_PCT.keys())
    vals = list(NCCL_KERNEL_PCT.values())
    cols = ["#F58518", "#E45756", "#54A24B"]
    ax.bar(labels, vals, color=cols, width=0.6)
    ax.set_ylabel("% of CUDA kernel time")
    style_ax(ax, "Nsight: dominant NCCL share")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
    ax.text(
        0.5,
        -0.22,
        "DP≈AllReduce grads · TP≈AllReduce bf16 · PP≈SendRecv",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#555555",
    )

    ax = axes[1]
    main = train[train["config"].isin(["1GPU", "DP2", "TP2", "PP2_mbs1"])]
    order = ["1GPU", "DP2", "TP2", "PP2_mbs1"]
    main = main.set_index("config").loc[order]
    x = np.arange(len(order))
    w = 0.35
    g0 = main["gpu0_avg_util"].astype(float).values
    g1 = main["gpu1_avg_util"].astype(float).values
    ax.bar(x - w / 2, g0, w, label="GPU0", color="#4C78A8")
    ax.bar(x + w / 2, g1, w, label="GPU1", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(["1GPU", "DP2", "TP2", "PP2\n(mbs=1)"])
    ax.set_ylabel("Avg SM util (%)")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax, "GPU utilization (nvidia-smi dmon)")

    fig.tight_layout()
    path = FIG / "fig5_comm_util.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_efficiency_memory_tradeoff(train: pd.DataFrame):
    main = train[train["config"].isin(["1GPU", "DP2", "TP2", "PP2_mbs1", "PP2_mbs4"])].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors_map = {
        "1GPU": "#4C78A8",
        "DP2": "#F58518",
        "TP2": "#E45756",
        "PP2_mbs1": "#54A24B",
        "PP2_mbs4": "#72B7B2",
    }
    for _, r in main.iterrows():
        ax.scatter(
            r["peak_alloc_mb"] / 1024,
            r["tokens_per_sec"] / 1000,
            s=160,
            color=colors_map[r["config"]],
            zorder=3,
        )
        ax.annotate(
            r["config"].replace("_", "\n"),
            (r["peak_alloc_mb"] / 1024, r["tokens_per_sec"] / 1000),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )
    ax.set_xlabel("Peak allocated VRAM per GPU (GB)")
    ax.set_ylabel("Throughput (k tokens/s)")
    style_ax(ax, "Throughput vs memory tradeoff")
    fig.tight_layout()
    path = FIG / "fig6_tradeoff.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_pdf(figs: dict, train: pd.DataFrame, nccl: pd.DataFrame, baseline: float):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleBig",
            parent=styles["Title"],
            fontSize=20,
            spaceAfter=8,
            textColor=colors.HexColor("#1a1a1a"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1c",
            parent=styles["Heading1"],
            fontSize=14,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#1f4e79"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2c",
            parent=styles["Heading2"],
            fontSize=11.5,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#2e75b6"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJ",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#444444"),
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletBody",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=12.5,
            leftIndent=8,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeSmall",
            parent=styles["Code"],
            fontSize=7.5,
            leading=9.5,
            backColor=colors.HexColor("#f4f4f4"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="FooterNote",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#666666"),
            alignment=TA_CENTER,
        )
    )

    def P(text, style="BodyJ"):
        return Paragraph(text, styles[style])

    def fig(path, width=6.7 * inch, caption=""):
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            w, h = im.size
        aspect = h / float(w)
        img = Image(str(path), width=width, height=width * aspect)
        flow = [img]
        if caption:
            flow.append(Paragraph(caption, styles["Caption"]))
        return KeepTogether(flow)

    def table(data, col_widths=None):
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("ALIGN", (0, 0), (0, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fb")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    one = train.loc[train["config"] == "1GPU"].iloc[0]
    dp = train.loc[train["config"] == "DP2"].iloc[0]
    tp = train.loc[train["config"] == "TP2"].iloc[0]
    pp1 = train.loc[train["config"] == "PP2_mbs1"].iloc[0]
    pp2 = train.loc[train["config"] == "PP2_mbs2"].iloc[0]
    pp4 = train.loc[train["config"] == "PP2_mbs4"].iloc[0]
    weak = train.loc[train["config"] == "DP2_weak"].iloc[0]

    dp_eff = float(dp["speedup"] / 2) * 100
    weak_eff = float(one["mean_step_ms"] / weak["mean_step_ms"]) * 100
    tp_mem_save = (1 - float(tp["peak_alloc_mb"]) / float(one["peak_alloc_mb"])) * 100
    pp_mem_save = (1 - float(pp1["peak_alloc_mb"]) / float(one["peak_alloc_mb"])) * 100

    story = []
    story.append(P("Single-Node 2×A100 Parallelism Experiment Report", "TitleBig"))
    story.append(
        P(
            "<b>DP vs TP vs PP on NVLink</b> — Megatron-LM mock-data training study<br/>"
            "Lab path: <font face='Courier'>/root/working/dp-tp-pp-lab</font> &nbsp;|&nbsp; "
            "Generated from measured logs (not synthetic estimates)",
            "FooterNote",
        )
    )
    story.append(Spacer(1, 8))

    # Executive summary
    story.append(P("1. Executive Summary", "H1c"))
    story.append(
        P(
            "This report compares <b>Data Parallel (DP=2)</b>, <b>Tensor Parallel (TP=2)</b>, and "
            "<b>Pipeline Parallel (PP=2)</b> against a single-GPU baseline on one Vast.ai node with "
            "<b>2× NVIDIA A100-SXM4-40GB</b> connected by <b>NV12</b> (12 NVLinks). "
            "All runs use the same GPT-like Megatron model, BF16, mock data, global batch 64 "
            "(except DP weak scaling), sequence length 2048, 20 warmup + 100 measured iterations."
        )
    )
    story.append(
        P(
            f"<b>Key findings.</b> DP2 delivers <b>{dp['speedup']:.2f}×</b> aggregate throughput "
            f"({dp_eff:.0f}% strong-scaling efficiency) with almost no communication overhead on NVLink. "
            f"TP2 is only <b>{tp['speedup']:.2f}×</b> faster than 1GPU because NCCL collectives consume "
            f"~{NCCL_KERNEL_PCT['TP2']:.0f}% of profiled CUDA kernel time, but it cuts peak VRAM by "
            f"~{tp_mem_save:.0f}%. PP2 reaches up to <b>{pp4['speedup']:.2f}×</b> at microbatch=4 as "
            "pipeline bubble shrinks relative to useful work and larger microbatches improve GEMM efficiency; "
            f"at microbatch=1 it already saves ~{pp_mem_save:.0f}% VRAM versus the full replica."
        )
    )

    # Setup
    story.append(P("2. Experimental Setup", "H1c"))
    story.append(P("2.1 Hardware & software", "H2c"))
    setup_tbl = table(
        [
            ["Item", "Value"],
            ["GPUs", "2 × NVIDIA A100-SXM4-40GB (40 GB each)"],
            ["Topology", "GPU0 ↔ GPU1 = NV12 (12× 25 GB/s NVLink)"],
            ["Driver / CUDA", "580.159.03 / toolkit 13.1"],
            ["PyTorch", "2.10.0a0+a36e1d39eb.nv26.01 (NGC)"],
            ["Megatron-LM", "commit be85fc5 (Megatron Core 0.20)"],
            ["Precision / data", "BF16 / --mock-data (NullTokenizer)"],
            ["Entry point", "pretrain_gpt.py via torch.distributed.run"],
        ],
        col_widths=[1.6 * inch, 5.0 * inch],
    )
    story.append(setup_tbl)
    story.append(Spacer(1, 8))

    story.append(P("2.2 Frozen model configuration", "H2c"))
    story.append(
        Preformatted(
            "num_layers=24  hidden_size=2048  ffn_hidden_size=8192  num_attention_heads=16\n"
            "seq_length=2048  vocab_size≈32000  micro_batch_size=1 (default)  global_batch_size=64\n"
            "train_iters=120 (warmup≈20, measured≈100)  optimizer=Adam  LR cosine 1.5e-4→1.5e-5",
            styles["CodeSmall"],
        )
    )
    story.append(
        P(
            "Tokens per optimizer step = global_batch × seq_len = <b>64 × 2048 = 131,072</b> "
            "(262,144 for DP weak scaling with global_batch=128). "
            "Throughput = tokens_per_step / mean_step_time (warmup excluded). "
            "The 1GPU configuration peaked at <b>~33.9 GB</b> allocated and fit without shrinking the model."
        )
    )

    story.append(P("2.3 What each parallelism mode does", "H2c"))
    story.append(
        P(
            "<b>DP=2:</b> each GPU holds a full model replica and trains on different microbatches; "
            "gradients are synchronized with AllReduce once per step.<br/>"
            "<b>TP=2:</b> one model replica is sharded across GPUs inside each Transformer layer; "
            "AllReduce/collectives run many times per layer during forward and backward.<br/>"
            "<b>PP=2:</b> layers are split into two stages (≈12 layers each); activations and gradients "
            "move between stages via P2P SendRecv; idle time appears as a pipeline bubble."
        )
    )

    # NCCL
    story.append(P("3. NVLink Communication Baseline (NCCL)", "H1c"))
    story.append(
        P(
            "Before training, raw GPU↔GPU bandwidth was measured with NVIDIA nccl-tests "
            "(AllReduce / AllGather / ReduceScatter, 8 MiB → 1 GiB, 2 GPUs). "
            "This establishes the physical capability of the NV12 link that DP/TP/PP will use."
        )
    )
    story.append(fig(figs["nccl"], caption="Figure 1. NCCL bus bandwidth vs message size on NV12."))
    ar = nccl[(nccl["collective"] == "all_reduce") & (nccl["size_bytes"] == 1073741824)].iloc[0]
    ag = nccl[(nccl["collective"] == "all_gather") & (nccl["size_bytes"] == 1073741824)].iloc[0]
    rs = nccl[(nccl["collective"] == "reduce_scatter") & (nccl["size_bytes"] == 1073741824)].iloc[0]
    story.append(
        table(
            [
                ["Collective (1 GiB)", "time (µs)", "algbw (GB/s)", "busbw (GB/s)", "Relevance"],
                ["AllReduce", f"{ar.time_us:.0f}", f"{ar.algbw_GBs:.1f}", f"{ar.busbw_GBs:.1f}", "DP grad sync"],
                ["AllGather", f"{ag.time_us:.0f}", f"{ag.algbw_GBs:.1f}", f"{ag.busbw_GBs:.1f}", "TP-style ops"],
                ["ReduceScatter", f"{rs.time_us:.0f}", f"{rs.algbw_GBs:.1f}", f"{rs.busbw_GBs:.1f}", "TP-style ops"],
            ],
            col_widths=[1.5 * inch, 1.0 * inch, 1.2 * inch, 1.2 * inch, 1.4 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "Observation: large-message AllReduce busbw ≈ <b>195 GB/s</b> on this pair — high enough that "
            "infrequent DP synchronization is cheap, but not free enough to hide TP’s many small/medium "
            "collectives inside every layer."
        )
    )

    story.append(PageBreak())

    # Main results
    story.append(P("4. Main Training Results", "H1c"))
    story.append(
        P(
            "All primary configs use identical architecture, sequence length, precision, and global batch 64. "
            "Metrics below are computed from Megatron iteration logs after discarding the first 20 warmup steps. "
            "Peak memory uses Megatron’s reported <font face='Courier'>max allocated</font>."
        )
    )

    main_rows = [
        ["Config", "GPUs", "TP", "PP", "DP", "Tok/s", "Speedup", "Tok/s/GPU", "Step ms", "Peak GB", "Util%"],
    ]
    for cfg, label in [
        ("1GPU", "1GPU"),
        ("DP2", "DP2"),
        ("TP2", "TP2"),
        ("PP2_mbs1", "PP2 mbs1"),
        ("PP2_mbs2", "PP2 mbs2"),
        ("PP2_mbs4", "PP2 mbs4"),
    ]:
        r = train.loc[train["config"] == cfg].iloc[0]
        util = f"{r.gpu0_avg_util:.0f}"
        if r.num_gpus > 1:
            util += f"/{r.gpu1_avg_util:.0f}"
        main_rows.append(
            [
                label,
                int(r.num_gpus),
                int(r.tp),
                int(r.pp),
                int(r.dp),
                f"{r.tokens_per_sec:.0f}",
                f"{r.speedup:.2f}×",
                f"{r.tokens_per_sec_per_gpu:.0f}",
                f"{r.mean_step_ms:.0f}",
                f"{r.peak_alloc_mb/1024:.1f}",
                util,
            ]
        )
    story.append(table(main_rows, col_widths=[0.85 * inch] + [0.48 * inch] * 4 + [0.7 * inch] * 5 + [0.6 * inch]))
    story.append(Spacer(1, 8))
    story.append(
        fig(
            figs["main"],
            caption="Figure 2. Throughput, speedup, and peak VRAM for 1GPU / DP2 / TP2 / PP2(mbs=1).",
        )
    )
    story.append(
        fig(
            figs["tradeoff"],
            width=5.8 * inch,
            caption="Figure 3. Throughput–memory tradeoff. DP maximizes tokens/s; TP minimizes VRAM; PP sits in between and can approach DP throughput with enough microbatches.",
        )
    )

    # DP
    story.append(P("5. Data Parallel (DP=2)", "H1c"))
    story.append(P("5.1 Strong scaling (same global batch)", "H2c"))
    story.append(
        P(
            f"With global_batch fixed at 64, DP2 reduces mean step time from "
            f"<b>{one.mean_step_ms:.0f} ms</b> to <b>{dp.mean_step_ms:.0f} ms</b> and raises throughput from "
            f"<b>{one.tokens_per_sec:.0f}</b> to <b>{dp.tokens_per_sec:.0f}</b> tokens/s "
            f"(<b>{dp.speedup:.2f}×</b>, efficiency <b>{dp_eff:.1f}%</b>). "
            "Each GPU still stores a full replica (~33.9 GB peak), so DP does <b>not</b> solve capacity limits; "
            "it replicates work across data shards."
        )
    )
    story.append(P("5.2 Weak scaling (2× global batch)", "H2c"))
    story.append(
        fig(
            figs["weak"],
            caption="Figure 4. DP strong vs weak scaling. Ideal weak scaling keeps step time flat when GPUs and work both ×2.",
        )
    )
    story.append(
        P(
            f"Raising global_batch to 128 on 2 GPUs yields step time <b>{weak.mean_step_ms:.0f} ms</b> vs "
            f"<b>{one.mean_step_ms:.0f} ms</b> on 1GPU — weak-scaling efficiency "
            f"<b>{weak_eff:.1f}%</b>. Aggregate tokens/s rises to <b>{weak.tokens_per_sec:.0f}</b>. "
            "This is the regime that usually extends cleanly to multi-node DP when interconnect is adequate."
        )
    )

    # TP
    story.append(P("6. Tensor Parallel (TP=2)", "H1c"))
    story.append(
        P(
            f"TP2 keeps one logical replica but shards tensors across GPUs. Throughput is "
            f"<b>{tp.tokens_per_sec:.0f}</b> tokens/s (<b>{tp.speedup:.2f}×</b> vs 1GPU) — faster than one GPU, "
            f"but far behind DP2. Peak allocated memory falls to <b>{tp.peak_alloc_mb/1024:.1f} GB</b> "
            f"(≈{tp_mem_save:.0f}% savings). GPU0 average SM util is lower ({tp.gpu0_avg_util:.0f}%) than GPU1 "
            f"({tp.gpu1_avg_util:.0f}%), consistent with communication/wait imbalance under frequent collectives."
        )
    )
    story.append(
        P(
            "<b>Why TP underperforms DP here:</b> the model already fits on one A100, so TP’s benefit is mostly "
            "memory headroom, not necessary for capacity. Every linear/attention partition pays NVLink collectives; "
            "Nsight shows AllReduce alone at ~37% of CUDA kernel time. DP pays AllReduce only once per step (~1.3%)."
        )
    )

    # PP
    story.append(P("7. Pipeline Parallel (PP=2)", "H1c"))
    story.append(
        P(
            f"PP2 splits the 24 layers into two stages. With microbatch=1 (64 microbatches/step), throughput is "
            f"<b>{pp1.tokens_per_sec:.0f}</b> tokens/s (<b>{pp1.speedup:.2f}×</b>) and peak memory "
            f"<b>{pp1.peak_alloc_mb/1024:.1f} GB</b> (≈{pp_mem_save:.0f}% vs 1GPU). "
            "Unlike TP, PP does not make a single layer faster; it overlaps different microbatches on different stages."
        )
    )
    story.append(
        fig(
            figs["pp"],
            caption="Figure 5. PP=2 microbatch sweep: throughput rises with larger microbatches while activation memory grows.",
        )
    )
    story.append(
        table(
            [
                ["micro_batch", "#microbatches", "Tok/s", "Speedup", "Step ms", "Peak GB"],
                ["1", "64", f"{pp1.tokens_per_sec:.0f}", f"{pp1.speedup:.2f}×", f"{pp1.mean_step_ms:.0f}", f"{pp1.peak_alloc_mb/1024:.1f}"],
                ["2", "32", f"{pp2.tokens_per_sec:.0f}", f"{pp2.speedup:.2f}×", f"{pp2.mean_step_ms:.0f}", f"{pp2.peak_alloc_mb/1024:.1f}"],
                ["4", "16", f"{pp4.tokens_per_sec:.0f}", f"{pp4.speedup:.2f}×", f"{pp4.mean_step_ms:.0f}", f"{pp4.peak_alloc_mb/1024:.1f}"],
            ],
            col_widths=[1.1 * inch, 1.2 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "Tradeoff: more microbatches reduce bubble fraction ≈ (PP−1)/N<sub>micro</sub>, while larger microbatches "
            "improve GEMM efficiency and raise activation memory. On this 2-stage pipeline, increasing microbatch "
            "from 1→4 raised tokens/s by "
            f"<b>{(pp4.tokens_per_sec/pp1.tokens_per_sec - 1)*100:.0f}%</b> even though bubble fraction grew "
            "(1/64 → 1/16), because compute intensity improved enough to dominate."
        )
    )

    story.append(PageBreak())

    # Nsight
    story.append(P("8. Nsight Systems Communication Evidence", "H1c"))
    story.append(
        P(
            "Short profiles were captured for DP2, TP2, and PP2 (see "
            "<font face='Courier'>profiles/nsys_*.nsys-rep</font>). "
            "Kernel summaries quantify how much GPU time is spent in NCCL versus math kernels."
        )
    )
    story.append(
        fig(
            figs["comm"],
            caption="Figure 6. Left: share of profiled CUDA kernel time in the dominant NCCL kernel. Right: average SM utilization during full benchmarks.",
        )
    )
    story.append(
        table(
            [
                ["Config", "Dominant NCCL kernel", "% kernel time", "Timeline intuition"],
                ["DP2", "AllReduce (f32 grads)", f"{NCCL_KERNEL_PCT['DP2']:.1f}%", "Long compute → short sync"],
                ["TP2", "AllReduce (bf16)", f"{NCCL_KERNEL_PCT['TP2']:.1f}%", "GEMM ↔ NCCL interleaved"],
                ["PP2", "SendRecv", f"{NCCL_KERNEL_PCT['PP2']:.1f}%", "Staggered stages + bubble"],
            ],
            col_widths=[0.8 * inch, 1.7 * inch, 1.1 * inch, 2.6 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        P(
            "<b>Takeaway:</b> communication frequency — not just peak NVLink bandwidth — decides efficiency. "
            "DP communicates rarely and scales well. TP communicates constantly and leaves throughput on the table "
            "even on NV12. PP communicates at stage boundaries and exposes idle bubbles that microbatching amortizes."
        )
    )

    # Q&A
    story.append(P("9. Answers to the Study Questions", "H1c"))
    qa = [
        (
            "1. How much faster is DP=2 than one GPU?",
            f"<b>{dp.speedup:.2f}×</b> aggregate tokens/s ({dp.tokens_per_sec:.0f} vs {one.tokens_per_sec:.0f}).",
        ),
        (
            "2. What is DP scaling efficiency?",
            f"Strong: <b>{dp_eff:.1f}%</b> (= speedup/2). Weak (GBS 64→128): <b>{weak_eff:.1f}%</b>.",
        ),
        (
            "3. For the same global batch, is TP=2 faster than one GPU?",
            f"<b>Yes, modestly</b> — {tp.speedup:.2f}× ({tp.tokens_per_sec:.0f} tokens/s).",
        ),
        (
            "4. Is TP=2 more or less efficient than DP=2?",
            f"<b>Less efficient</b> ({tp.speedup:.2f}× vs {dp.speedup:.2f}×; {tp.tokens_per_sec_per_gpu:.0f} vs {dp.tokens_per_sec_per_gpu:.0f} tok/s/GPU).",
        ),
        (
            "5. How much VRAM does TP save?",
            f"Peak allocated <b>{one.peak_alloc_mb/1024:.1f} → {tp.peak_alloc_mb/1024:.1f} GB</b> (~{tp_mem_save:.0f}% reduction).",
        ),
        (
            "6. How much VRAM does PP save?",
            f"At mbs=1: <b>{one.peak_alloc_mb/1024:.1f} → {pp1.peak_alloc_mb/1024:.1f} GB</b> (~{pp_mem_save:.0f}%). "
            f"Savings shrink as microbatch grows (mbs=4 → {pp4.peak_alloc_mb/1024:.1f} GB).",
        ),
        (
            "7. How does PP throughput change with microbatches?",
            f"mbs 1→2→4: <b>{pp1.tokens_per_sec:.0f} → {pp2.tokens_per_sec:.0f} → {pp4.tokens_per_sec:.0f}</b> tokens/s.",
        ),
        (
            "8. What do DP/TP/PP look like in Nsight?",
            "DP: compute-heavy with rare AllReduce. TP: dense AllReduce between GEMMs (~37% kernel time). "
            "PP: SendRecv between offset stage timelines with bubble/idle regions.",
        ),
        (
            "9. Which strategy communicates most frequently?",
            "<b>TP</b> (tens of thousands of AllReduce instances in the profile window).",
        ),
        (
            "10. Why prefer DP when the model fits on one GPU?",
            "Highest tokens/s with minimal communication; nearly ideal scaling on NVLink in this study.",
        ),
        (
            "11. Why might TP still be useful if the model fits?",
            "Halves per-GPU parameter memory, enables larger batches/models later, and composes with DP/PP — "
            "at a clear throughput cost from collectives.",
        ),
        (
            "12. Why is PP mainly useful for larger models / more GPUs?",
            "It stages layers so models that exceed single-GPU memory can train; deeper pipelines need many "
            "microbatches (or virtual stages) to keep bubbles small.",
        ),
    ]
    for q, a in qa:
        story.append(P(f"<b>{q}</b><br/>{a}", "BulletBody"))

    # Conclusions
    story.append(P("10. Conclusions & Practical Guidance", "H1c"))
    story.append(
        P(
            "On this <b>2×A100 NV12</b> node with a model that already fits in 40 GB:"
        )
    )
    story.append(
        P(
            "• Choose <b>DP</b> for maximum training throughput and simplest scaling.<br/>"
            "• Choose <b>TP</b> when you need per-GPU memory headroom or are building multi-dimensional parallel plans; "
            "expect heavy collective traffic even on NVLink.<br/>"
            "• Choose <b>PP</b> to split depth for capacity; tune microbatch size — too small leaves bubble/GEMM inefficiency, "
            "too large blows activation memory.<br/>"
            "• Always measure NCCL first: good NVLink bandwidth is necessary but not sufficient if your parallel strategy "
            "communicates every layer."
        )
    )

    story.append(P("11. Reproducibility", "H1c"))
    story.append(
        Preformatted(
            "cd /root/working/dp-tp-pp-lab\n"
            "bash run_all.sh\n"
            "# artifacts: results/*.csv  logs/*.log  profiles/nsys_*.nsys-rep  analysis/figures/\n"
            "python3 analysis/generate_report.py   # regenerates this PDF",
            styles["CodeSmall"],
        )
    )
    story.append(Spacer(1, 12))
    story.append(
        P(
            "End of report — all numeric claims are taken from "
            "<font face='Courier'>results/training_results.csv</font>, "
            "<font face='Courier'>results/nccl_results.csv</font>, Megatron logs, and Nsight kernel summaries.",
            "FooterNote",
        )
    )

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(20 * mm, 12 * mm, "DP/TP/PP Lab — 2×A100 NV12")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="2×A100 DP/TP/PP Experiment Report",
        author="dp-tp-pp-lab",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {OUT_PDF}")


def main():
    train, nccl, baseline = load_data()
    figs = {
        "main": plot_main_comparison(train),
        "pp": plot_pp_sweep(train),
        "weak": plot_weak_scaling(train),
        "nccl": plot_nccl(nccl),
        "comm": plot_comm_and_util(train),
        "tradeoff": plot_efficiency_memory_tradeoff(train),
    }
    print("Figures:", {k: str(v) for k, v in figs.items()})
    build_pdf(figs, train, nccl, baseline)


if __name__ == "__main__":
    main()
