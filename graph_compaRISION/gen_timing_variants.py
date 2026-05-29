#!/usr/bin/env python3
"""Generate 20+ timing-breakdown figure variants for a CoRL paper figure.

Designed for inclusion at width=0.5\\columnwidth (~1.69").
Figures are rendered at 3.4" wide so fonts appear at 2x their final point
size — a 16pt label becomes ~8pt when LaTeX scales the figure to half-column.
Run:  python gen_timing_variants.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    print("seaborn not found — skipping seaborn variants")

OUT_DIR = Path(__file__).parent / "outputs_timing_variants"
OUT_DIR.mkdir(exist_ok=True)

# ── Shared data ──────────────────────────────────────────────────────────────
METHODS        = ["Geometric\nPipeline", "E2E\n(ViT-b)", "E2E\n(ViT-s)"]
METHODS_LONG   = ["Geometric Pipeline", "E2E (ViT-b)", "E2E (ViT-s)"]

BASELINE_STEPS = ["parse cloud", "elev. map", "local graph", "global merge", "overhead"]
E2E_STEPS      = ["inference", "merge", "edge build", "other"]

BASELINE_VALS  = [2.4, 3.0, 6.4, 2.8, 7.1]
VITB_VALS      = [23.3, 0.8, 0.9, 0.7]
VITS_VALS      = [12.4, 0.7, 0.9, 0.7]

ALL_DATA = [
    (METHODS[0], BASELINE_STEPS, BASELINE_VALS),
    (METHODS[1], E2E_STEPS,      VITB_VALS),
    (METHODS[2], E2E_STEPS,      VITS_VALS),
]
TOTALS = [sum(v) for _, _, v in ALL_DATA]

# Same palette everywhere — bottom-to-top segment colors
PAL = ["#e19c24", "#8fb032", "#d9ecf9", "#f2e9b9", "#e3c05d"]
def pc(j): return PAL[j % len(PAL)]

METHOD_COLORS = ["#4e79a7", "#f28e2b", "#59a14f"]

# Figure width that, when included at 0.5\columnwidth, scales to ~1.69"
W = 3.4   # inches

def save(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {name}.png")

# ─────────────────────────────────────────────────────────────────────────────
# v01 — vertical stacked bars, bold totals above, legend below, no in-bar text
# ─────────────────────────────────────────────────────────────────────────────
def v01():
    fig, ax = plt.subplots(figsize=(W, 4.2))
    xs = np.arange(len(ALL_DATA))
    bw = 0.55
    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            ax.bar(xs[mi], v, bottom=cum, width=bw, color=pc(j),
                   edgecolor="white", linewidth=0.6)
            cum += v
        ax.text(xs[mi], cum + 0.6, f"{cum:.1f} ms",
                ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(METHODS, fontsize=13)
    ax.set_ylabel("Mean time per frame (ms)", fontsize=13)
    ax.set_ylim(0, max(TOTALS) * 1.28)
    ax.grid(axis="y", alpha=0.35, linewidth=0.6)
    ax.spines[["top","right"]].set_visible(False)

    all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
    handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
    ax.legend(handles=handles, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "v01")

# ─────────────────────────────────────────────────────────────────────────────
# v02 — horizontal stacked bars, labels outside right
# ─────────────────────────────────────────────────────────────────────────────
def v02():
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ys   = np.arange(len(ALL_DATA))
    bh   = 0.52
    xmax = max(TOTALS) * 1.42

    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            ax.barh(ys[mi], v, left=cum, height=bh, color=pc(j),
                    edgecolor="white", linewidth=0.6)
            cum += v
        ax.text(cum + 0.4, ys[mi], f"{cum:.1f} ms",
                ha="left", va="center", fontsize=13, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels(METHODS, fontsize=13)
    ax.set_xlabel("Mean time per frame (ms)", fontsize=12)
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", alpha=0.35, linewidth=0.6)
    ax.spines[["top","right"]].set_visible(False)

    all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
    handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
    ax.legend(handles=handles, fontsize=8, ncol=1, loc="lower right", frameon=False)
    save(fig, "v02")

# ─────────────────────────────────────────────────────────────────────────────
# v03 — horizontal stacked, step labels INSIDE (abbreviated), no legend
# ─────────────────────────────────────────────────────────────────────────────
def v03():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ys = np.arange(len(ALL_DATA))
    bh = 0.55
    SHORT = {"parse cloud":"parse", "elev. map":"emap", "local graph":"local",
             "global merge":"merge", "overhead":"ovhd",
             "inference":"infer", "edge build":"edges", "other":"other"}

    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            bar = ax.barh(ys[mi], v, left=cum, height=bh, color=pc(j),
                          edgecolor="white", linewidth=0.5)
            # only label segments wide enough to fit
            if v > 1.5:
                short = SHORT.get(s, s[:5])
                ax.text(cum + v / 2, ys[mi], f"{short}\n{v:.1f}", fontsize=9,
                        ha="center", va="center", color="black", linespacing=1.2)
            cum += v
        ax.text(cum + 0.3, ys[mi], f"{cum:.1f} ms",
                ha="left", va="center", fontsize=12, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels(METHODS, fontsize=13)
    ax.set_xlabel("Mean time per frame (ms)", fontsize=12)
    ax.set_xlim(0, max(TOTALS) * 1.38)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, "v03")

# ─────────────────────────────────────────────────────────────────────────────
# v04 — 100 % normalized horizontal stacked, step % labels inside
# ─────────────────────────────────────────────────────────────────────────────
def v04():
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ys = np.arange(len(ALL_DATA))
    bh = 0.55

    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        tot = sum(vals)
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            pct = v / tot * 100
            ax.barh(ys[mi], pct, left=cum, height=bh, color=pc(j),
                    edgecolor="white", linewidth=0.5)
            if pct > 7:
                ax.text(cum + pct/2, ys[mi], f"{pct:.0f}%",
                        ha="center", va="center", fontsize=11, fontweight="bold")
            cum += pct

    ax.set_yticks(ys)
    ax.set_yticklabels(METHODS, fontsize=13)
    ax.set_xlabel("Proportion of frame time (%)", fontsize=12)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)

    all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
    handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.38), frameon=False)
    fig.subplots_adjust(bottom=0.28)
    save(fig, "v04")

# ─────────────────────────────────────────────────────────────────────────────
# v05 — grouped bars (one bar per step per method), no stacking
# ─────────────────────────────────────────────────────────────────────────────
def v05():
    fig, ax = plt.subplots(figsize=(W, 3.8))
    all_steps = list(dict.fromkeys(BASELINE_STEPS + E2E_STEPS))
    n = len(all_steps)
    xs = np.arange(n)
    w  = 0.25
    offsets = [-w, 0, w]

    padded = []
    for _, steps, vals in ALL_DATA:
        d = dict(zip(steps, vals))
        padded.append([d.get(s, 0) for s in all_steps])

    for mi, (vals_pad, color) in enumerate(zip(padded, METHOD_COLORS)):
        ax.bar(xs + offsets[mi], vals_pad, width=w, color=color, alpha=0.9,
               edgecolor="white", linewidth=0.4, label=METHODS_LONG[mi])

    ax.set_xticks(xs)
    ax.set_xticklabels(all_steps, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Mean time (ms)", fontsize=12)
    ax.grid(axis="y", alpha=0.35, linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.28)
    save(fig, "v05")

# ─────────────────────────────────────────────────────────────────────────────
# v06 — lollipop chart: each method as a horizontal lollipop per step
# ─────────────────────────────────────────────────────────────────────────────
def v06():
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.6), sharey=False)
    for mi, (ax, (lbl, steps, vals), color) in enumerate(zip(axes, ALL_DATA, METHOD_COLORS)):
        ys = np.arange(len(steps))
        ax.hlines(ys, 0, vals, color=color, linewidth=2.0, alpha=0.8)
        ax.plot(vals, ys, "o", color=color, ms=6)
        for yi, v in zip(ys, vals):
            ax.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=10, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels(steps, fontsize=9)
        ax.set_xlim(0, max(vals) * 1.6)
        ax.set_title(lbl.replace("\n", " "), fontsize=11, fontweight="bold", pad=4)
        ax.set_xlabel("ms", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    fig.tight_layout(pad=0.6)
    save(fig, "v06")

# ─────────────────────────────────────────────────────────────────────────────
# v07 — seaborn whitegrid horizontal stacked
# ─────────────────────────────────────────────────────────────────────────────
def v07():
    if not HAS_SNS:
        return
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(W, 2.8))
        ys = np.arange(len(ALL_DATA))
        bh = 0.52
        for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
            cum = 0.0
            for j, (s, v) in enumerate(zip(steps, vals)):
                ax.barh(ys[mi], v, left=cum, height=bh, color=pc(j),
                        edgecolor="white", linewidth=0.6)
                if v > 2:
                    ax.text(cum + v/2, ys[mi], f"{v:.1f}", ha="center", va="center",
                            fontsize=11, fontweight="bold")
                cum += v
            ax.text(cum + 0.4, ys[mi], f"{cum:.1f} ms",
                    ha="left", va="center", fontsize=12, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels(METHODS, fontsize=13)
        ax.set_xlabel("Mean time per frame (ms)", fontsize=12)
        ax.set_xlim(0, max(TOTALS)*1.38)
        ax.spines[["top","right"]].set_visible(False)
        all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
        handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
        ax.legend(handles=handles, fontsize=8, ncol=1, loc="lower right", frameon=False)
    save(fig, "v07")

# ─────────────────────────────────────────────────────────────────────────────
# v08 — seaborn dark + vertical stacked + no in-bar text + big bold totals
# ─────────────────────────────────────────────────────────────────────────────
def v08():
    if not HAS_SNS:
        return
    with sns.axes_style("dark"):
        fig, ax = plt.subplots(figsize=(W, 4.0), facecolor="#1c1c2e")
        ax.set_facecolor("#1c1c2e")
        xs = np.arange(len(ALL_DATA))
        bw = 0.55
        DARK_PAL = ["#FFB347", "#77DD77", "#AEC6CF", "#FDFD96", "#CB99C9", "#FF6961", "#B5EAD7", "#C7CEEA", "#F0B27A"]
        all_steps_dark = list(dict.fromkeys(BASELINE_STEPS + E2E_STEPS))
        for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
            cum = 0.0
            for j, (s, v) in enumerate(zip(steps, vals)):
                gi = all_steps_dark.index(s)
                ax.bar(xs[mi], v, bottom=cum, width=bw, color=DARK_PAL[gi],
                       edgecolor="#1c1c2e", linewidth=0.8, alpha=0.92)
                cum += v
            ax.text(xs[mi], cum + 0.5, f"{cum:.1f} ms",
                    ha="center", va="bottom", fontsize=14, fontweight="bold", color="white")
        ax.set_xticks(xs)
        ax.set_xticklabels(METHODS, fontsize=13, color="white")
        ax.set_ylabel("Mean time per frame (ms)", fontsize=12, color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444466")
        ax.set_ylim(0, max(TOTALS)*1.28)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5, color="white")
        handles = [mpatches.Patch(color=DARK_PAL[i], label=all_steps_dark[i])
                   for i in range(len(all_steps_dark))]
        ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), frameon=False, labelcolor="white")
        fig.subplots_adjust(bottom=0.22)
    save(fig, "v08")

# ─────────────────────────────────────────────────────────────────────────────
# v09 — donut charts: one per method, step proportion shown
# ─────────────────────────────────────────────────────────────────────────────
def v09():
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.4))
    for ax, (lbl, steps, vals), tot in zip(axes, ALL_DATA, TOTALS):
        colors = [pc(j) for j in range(len(steps))]
        wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                           wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 0.7})
        ax.set_title(lbl.replace("\n", " "), fontsize=10, fontweight="bold", pad=3)
        ax.text(0, 0, f"{tot:.0f}\nms", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#222")
    # shared legend
    all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
    handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
    fig.legend(handles=handles, fontsize=7, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.subplots_adjust(bottom=0.25, wspace=0.1)
    save(fig, "v09")

# ─────────────────────────────────────────────────────────────────────────────
# v10 — vertical stacked, step labels outside bars (right column of text)
# ─────────────────────────────────────────────────────────────────────────────
def v10():
    fig, ax = plt.subplots(figsize=(W, 4.4))
    xs  = np.arange(len(ALL_DATA))
    bw  = 0.45
    xmax = len(ALL_DATA) - 1 + 1.35

    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            ax.bar(xs[mi], v, bottom=cum, width=bw, color=pc(j),
                   edgecolor="white", linewidth=0.6)
            # draw a small tick line + label to the right of the last bar
            mid_y = cum + v / 2
            if mi == len(ALL_DATA) - 1:
                ax.annotate(f"{s}  {v:.1f} ms",
                            xy=(xs[mi] + bw/2, mid_y),
                            xytext=(xs[mi] + bw/2 + 0.08, mid_y),
                            fontsize=8.5, va="center",
                            arrowprops=dict(arrowstyle="-", color="#777", lw=0.6))
            cum += v
        ax.text(xs[mi], cum + 0.6, f"{cum:.1f} ms",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(METHODS, fontsize=12)
    ax.set_ylabel("Mean time per frame (ms)", fontsize=12)
    ax.set_xlim(-0.6, xmax)
    ax.set_ylim(0, max(TOTALS)*1.22)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    save(fig, "v10")

# ─────────────────────────────────────────────────────────────────────────────
# v11 — seaborn ticks style, vertical stacked, total labels + bottom legend
# ─────────────────────────────────────────────────────────────────────────────
def v11():
    if not HAS_SNS:
        return
    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(figsize=(W, 4.0))
        xs = np.arange(len(ALL_DATA))
        bw = 0.55
        for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
            cum = 0.0
            for j, (s, v) in enumerate(zip(steps, vals)):
                ax.bar(xs[mi], v, bottom=cum, width=bw, color=pc(j),
                       edgecolor="white", linewidth=0.7)
                cum += v
            ax.text(xs[mi], cum + 0.5, f"{cum:.1f} ms",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(METHODS, fontsize=13)
        ax.set_ylabel("Mean time per frame (ms)", fontsize=12)
        ax.set_ylim(0, max(TOTALS)*1.26)
        sns.despine()
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
        handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
        ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.1), frameon=False)
        fig.subplots_adjust(bottom=0.22)
    save(fig, "v11")

# ─────────────────────────────────────────────────────────────────────────────
# v12 — two-panel: bar chart left, data table right
# ─────────────────────────────────────────────────────────────────────────────
def v12():
    fig = plt.figure(figsize=(W, 3.8))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.05)
    ax_bar = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    xs = np.arange(len(ALL_DATA))
    bw = 0.55
    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            ax_bar.bar(xs[mi], v, bottom=cum, width=bw, color=pc(j),
                       edgecolor="white", linewidth=0.6)
            cum += v
        ax_bar.text(xs[mi], cum + 0.5, f"{cum:.1f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(["Geom.", "ViT-b", "ViT-s"], fontsize=12)
    ax_bar.set_ylabel("ms", fontsize=11)
    ax_bar.set_ylim(0, max(TOTALS)*1.28)
    ax_bar.spines[["top","right"]].set_visible(False)
    ax_bar.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Table
    ax_tbl.axis("off")
    cols  = ["Geom.", "ViT-b", "ViT-s"]
    SHORT = {"parse cloud":"parse", "elev. map":"emap", "local graph":"local",
             "global merge":"merge", "overhead":"ovhd",
             "inference":"infer", "edge build":"edges", "other":"other"}
    all_steps = list(dict.fromkeys(BASELINE_STEPS + E2E_STEPS))
    all_data_map = [dict(zip(s, v)) for _, s, v in ALL_DATA]
    rows = [[SHORT.get(s, s)] + [f"{d.get(s,0):.1f}" for d in all_data_map]
            for s in all_steps]
    tbl = ax_tbl.table(cellText=rows, colLabels=["step"] + cols,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.12)
    # color header and step column cells
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#ddd")
        elif c == 0:
            si = r - 1
            cell.set_facecolor(pc(si) + "88")
        else:
            cell.set_facecolor("white")
    save(fig, "v12")

# ─────────────────────────────────────────────────────────────────────────────
# v13 — tall narrow vertical stacked, step labels as y-tick style on right
# ─────────────────────────────────────────────────────────────────────────────
def v13():
    fig, ax = plt.subplots(figsize=(W, 5.2))
    xs = np.arange(len(ALL_DATA))
    bw = 0.5

    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            ax.bar(xs[mi], v, bottom=cum, width=bw, color=pc(j),
                   edgecolor="white", linewidth=0.7)
            # ms value inside each segment (only if tall enough)
            if v > 1.2:
                ax.text(xs[mi], cum + v/2, f"{v:.1f}",
                        ha="center", va="center", fontsize=11, fontweight="bold", color="#333")
            cum += v
        ax.text(xs[mi], cum + 0.6, f"Total\n{cum:.1f} ms",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(METHODS, fontsize=14)
    ax.set_ylabel("Mean time per frame (ms)", fontsize=13)
    ax.set_ylim(0, max(TOTALS)*1.32)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
    handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
    ax.legend(handles=handles, fontsize=8.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.subplots_adjust(bottom=0.20)
    save(fig, "v13")

# ─────────────────────────────────────────────────────────────────────────────
# v14 — stacked bars + hatching, no color needed (grayscale-friendly)
# ─────────────────────────────────────────────────────────────────────────────
def v14():
    HATCHES = ["", "//", "xx", "..", "\\\\", "++", "oo"]
    fig, ax = plt.subplots(figsize=(W, 4.2))
    xs = np.arange(len(ALL_DATA))
    bw = 0.55

    all_steps = list(dict.fromkeys(BASELINE_STEPS + E2E_STEPS))
    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            gi = all_steps.index(s)
            ax.bar(xs[mi], v, bottom=cum, width=bw, color=pc(gi),
                   edgecolor="#333", linewidth=0.5, hatch=HATCHES[gi % len(HATCHES)])
            cum += v
        ax.text(xs[mi], cum + 0.5, f"{cum:.1f} ms",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(METHODS, fontsize=13)
    ax.set_ylabel("Mean time per frame (ms)", fontsize=12)
    ax.set_ylim(0, max(TOTALS)*1.28)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.35, linewidth=0.5)
    handles = [mpatches.Patch(color=pc(i), hatch=HATCHES[i % len(HATCHES)],
                              edgecolor="#333", label=all_steps[i])
               for i in range(len(all_steps))]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "v14")

# ─────────────────────────────────────────────────────────────────────────────
# v15 — clean minimal: large total numbers, thin coloured bars, no grid
# ─────────────────────────────────────────────────────────────────────────────
def v15():
    fig, ax = plt.subplots(figsize=(W, 3.6))
    xs = np.arange(len(ALL_DATA))
    bw = 0.25   # thin bars

    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            ax.bar(xs[mi], v, bottom=cum, width=bw, color=pc(j),
                   edgecolor="none", linewidth=0)
            cum += v
        ax.text(xs[mi], cum + 1.0, f"{cum:.1f}",
                ha="center", va="bottom", fontsize=22, fontweight="bold", color="#222")
        ax.text(xs[mi], cum + 1.0 + 2.2, "ms",
                ha="center", va="bottom", fontsize=12, color="#666")

    ax.set_xticks(xs)
    ax.set_xticklabels(METHODS, fontsize=13)
    ax.set_ylim(0, max(TOTALS) * 1.45)
    ax.axis("off")
    ax.set_xticks(xs)   # re-enable x ticks after axis off
    for xi, lbl in zip(xs, METHODS):
        ax.text(xi, -2.5, lbl.replace("\n", "\n"), ha="center", va="top",
                fontsize=13, color="#333")
    all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
    handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 0.0), frameon=False)
    save(fig, "v15")

# ─────────────────────────────────────────────────────────────────────────────
# v16 — seaborn paper context, horizontal stacked, ms labels inside big segs
# ─────────────────────────────────────────────────────────────────────────────
def v16():
    if not HAS_SNS:
        return
    sns.set_context("paper", font_scale=1.6)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(W, 2.8))
        ys = np.arange(len(ALL_DATA))
        bh = 0.58
        for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
            cum = 0.0
            for j, (s, v) in enumerate(zip(steps, vals)):
                ax.barh(ys[mi], v, left=cum, height=bh, color=pc(j),
                        edgecolor="white", linewidth=0.7)
                if v >= 3:
                    ax.text(cum + v/2, ys[mi], f"{v:.1f}",
                            ha="center", va="center", fontsize=11, fontweight="bold")
                cum += v
            ax.text(cum + 0.4, ys[mi], f"{cum:.1f} ms",
                    ha="left", va="center", fontsize=13, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels(METHODS, fontsize=13)
        ax.set_xlabel("Mean time per frame (ms)", fontsize=12)
        ax.set_xlim(0, max(TOTALS)*1.35)
        ax.spines[["top","right"]].set_visible(False)
        all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
        handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
        ax.legend(handles=handles, fontsize=8, ncol=1, loc="lower right", frameon=False)
    sns.set_context("notebook")   # reset
    save(fig, "v16")

# ─────────────────────────────────────────────────────────────────────────────
# v17 — small multiples: one subplot per method, horizontal bar per step
# ─────────────────────────────────────────────────────────────────────────────
def v17():
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.8), sharey=False)
    for ax, (lbl, steps, vals), mcolor in zip(axes, ALL_DATA, METHOD_COLORS):
        ys = np.arange(len(steps))
        colors = [pc(j) for j in range(len(steps))]
        bars = ax.barh(ys, vals, color=colors, edgecolor="white", linewidth=0.5, height=0.65)
        for yi, v in zip(ys, vals):
            ax.text(v + 0.2, yi, f"{v:.1f}", va="center", fontsize=9.5, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels(steps, fontsize=8.5)
        ax.set_xlim(0, max(vals)*1.65)
        ax.set_title(lbl.replace("\n"," "), fontsize=10, fontweight="bold", pad=3, color=mcolor)
        ax.set_xlabel("ms", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linewidth=0.4)
    fig.tight_layout(pad=0.5)
    save(fig, "v17")

# ─────────────────────────────────────────────────────────────────────────────
# v18 — step-comparison small multiples: one row per step, bar per method
# ─────────────────────────────────────────────────────────────────────────────
def v18():
    all_steps = list(dict.fromkeys(BASELINE_STEPS + E2E_STEPS))
    n = len(all_steps)
    fig, axes = plt.subplots(1, n, figsize=(W, 2.6), sharey=False)
    all_data_map = [dict(zip(s, v)) for _, s, v in ALL_DATA]
    for ax, step, ji in zip(axes, all_steps, range(n)):
        vals = [d.get(step, 0) for d in all_data_map]
        bars = ax.bar(np.arange(3), vals, color=[METHOD_COLORS[i] for i in range(3)],
                      edgecolor="white", linewidth=0.4, width=0.7)
        ax.set_title(step, fontsize=7.5, fontweight="bold", pad=2,
                     rotation=45, ha="right", va="bottom")
        ax.set_xticks([])
        ax.set_ylim(0, max(vals + [0.5])*1.4)
        ax.spines[["top","right","bottom"]].set_visible(False)
        ax.tick_params(left=True, labelsize=8)
        ax.set_ylabel("ms" if ji == 0 else "", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, v + max(vals)*0.04, f"{v:.1f}", ha="center",
                        va="bottom", fontsize=7.5, fontweight="bold")
    handles = [mpatches.Patch(color=METHOD_COLORS[i], label=METHODS_LONG[i]) for i in range(3)]
    fig.legend(handles=handles, fontsize=7.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.subplots_adjust(bottom=0.22, wspace=0.5)
    save(fig, "v18")

# ─────────────────────────────────────────────────────────────────────────────
# v19 — seaborn paper context, vertical stacked, hatching, whitegrid
# ─────────────────────────────────────────────────────────────────────────────
def v19():
    if not HAS_SNS:
        return
    HATCHES = ["", "//", "xx", "..", "\\\\", "++", "oo"]
    sns.set_context("paper", font_scale=1.5)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(W, 4.2))
        xs = np.arange(len(ALL_DATA))
        bw = 0.55
        all_steps = list(dict.fromkeys(BASELINE_STEPS + E2E_STEPS))
        for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
            cum = 0.0
            for j, (s, v) in enumerate(zip(steps, vals)):
                gi = all_steps.index(s)
                ax.bar(xs[mi], v, bottom=cum, width=bw, color=pc(gi),
                       edgecolor="#333", linewidth=0.5, hatch=HATCHES[gi % len(HATCHES)])
                cum += v
            ax.text(xs[mi], cum + 0.5, f"{cum:.1f} ms",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(METHODS, fontsize=13)
        ax.set_ylabel("Mean time per frame (ms)", fontsize=12)
        ax.set_ylim(0, max(TOTALS)*1.28)
        sns.despine(left=False, bottom=False)
        handles = [mpatches.Patch(color=pc(i), hatch=HATCHES[i % len(HATCHES)],
                                  edgecolor="#333", label=all_steps[i])
                   for i in range(len(all_steps))]
        ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.1), frameon=False)
        fig.subplots_adjust(bottom=0.22)
    sns.set_context("notebook")
    save(fig, "v19")

# ─────────────────────────────────────────────────────────────────────────────
# v20 — diverging from E2E ViT-s baseline: show speedup/slowdown
# ─────────────────────────────────────────────────────────────────────────────
def v20():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ref   = TOTALS[2]   # ViT-s as reference
    diffs = [t - ref for t in TOTALS]
    colors = ["#e19c24" if d >= 0 else "#8fb032" for d in diffs]

    bars = ax.bar(np.arange(len(ALL_DATA)), diffs, color=colors,
                  edgecolor="white", linewidth=0.6, width=0.55)
    ax.axhline(0, color="#333", linewidth=1.0)
    for xi, (d, t) in enumerate(zip(diffs, TOTALS)):
        va  = "bottom" if d >= 0 else "top"
        off = 0.3 if d >= 0 else -0.3
        ax.text(xi, d + off, f"{t:.1f} ms\n({d:+.1f})",
                ha="center", va=va, fontsize=11, fontweight="bold")

    ax.set_xticks(np.arange(len(ALL_DATA)))
    ax.set_xticklabels(METHODS, fontsize=13)
    ax.set_ylabel("Δ from E2E ViT-s (ms)", fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_ylim(min(diffs) - 4, max(diffs) + 5)
    save(fig, "v20")

# ─────────────────────────────────────────────────────────────────────────────
# v21 — waterfall decomposition per method (sequential cumulative bars)
# ─────────────────────────────────────────────────────────────────────────────
def v21():
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.4), sharey=True)
    for ax, (lbl, steps, vals), mcolor in zip(axes, ALL_DATA, METHOD_COLORS):
        n  = len(steps)
        xs = np.arange(n)
        cum = 0.0
        starts = []
        for v in vals:
            starts.append(cum)
            cum += v
        bars = ax.bar(xs, vals, bottom=starts,
                      color=[pc(j) for j in range(n)],
                      edgecolor="white", linewidth=0.6, width=0.7)
        # connector lines
        for i in range(n - 1):
            ax.plot([xs[i] + 0.35, xs[i+1] - 0.35],
                    [starts[i] + vals[i], starts[i] + vals[i]],
                    color="#777", lw=0.7, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(steps, fontsize=7.5, rotation=35, ha="right")
        ax.set_title(lbl.replace("\n"," "), fontsize=10, fontweight="bold",
                     pad=3, color=mcolor)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    axes[0].set_ylabel("Cumulative time (ms)", fontsize=11)
    fig.tight_layout(pad=0.5)
    save(fig, "v21")

# ─────────────────────────────────────────────────────────────────────────────
# v22 — seaborn paper style, horizontal, big method names, step % in legend
# ─────────────────────────────────────────────────────────────────────────────
def v22():
    if not HAS_SNS:
        return
    sns.set_context("paper", font_scale=1.4)
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(W, 3.2))
        ys = np.arange(len(ALL_DATA))
        bh = 0.6
        for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
            cum = 0.0
            for j, (s, v) in enumerate(zip(steps, vals)):
                ax.barh(ys[mi], v, left=cum, height=bh, color=pc(j),
                        edgecolor="white", linewidth=0.8)
                cum += v
            # Bold total, tight to bar
            ax.text(cum + 0.2, ys[mi], f"{cum:.1f} ms",
                    ha="left", va="center", fontsize=13, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels(METHODS_LONG, fontsize=12)
        ax.set_xlabel("Mean time per frame (ms)", fontsize=12)
        ax.set_xlim(0, max(TOTALS) * 1.32)
        sns.despine()
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
        handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
        ax.legend(handles=handles, fontsize=8, ncol=2, loc="lower center",
                  bbox_to_anchor=(0.5, -0.38), frameon=False)
        fig.subplots_adjust(bottom=0.30)
    sns.set_context("notebook")
    save(fig, "v22")

# ─────────────────────────────────────────────────────────────────────────────
# v23 — Marimekko / mosaic: bar WIDTH encodes total time, height = proportion
# ─────────────────────────────────────────────────────────────────────────────
def v23():
    fig, ax = plt.subplots(figsize=(W, 3.8))
    max_total = max(TOTALS)
    x = 0.0
    gap = 0.8

    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        tot = sum(vals)
        bw  = tot / max_total * 2.5   # width proportional to total
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            h = v / tot * 100   # height = % of this method's total
            ax.bar(x + bw/2, h, bottom=cum, width=bw, color=pc(j),
                   edgecolor="white", linewidth=0.5, align="center")
            if h > 7:
                ax.text(x + bw/2, cum + h/2, f"{h:.0f}%",
                        ha="center", va="center", fontsize=9.5, fontweight="bold")
            cum += h
        ax.text(x + bw/2, 102, f"{tot:.1f} ms",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(x + bw/2, -7, lbl, ha="center", va="top", fontsize=11)
        x += bw + gap

    ax.set_xlim(-0.3, x - gap + 0.3)
    ax.set_ylim(-12, 115)
    ax.set_ylabel("Step share (%)", fontsize=12)
    ax.set_xlabel("← bar width = total time →", fontsize=10, color="#666")
    ax.set_xticks([])
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
    handles = [mpatches.Patch(color=pc(i), label=all_steps[i]) for i in range(len(all_steps))]
    ax.legend(handles=handles, fontsize=7.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.subplots_adjust(bottom=0.20)
    save(fig, "v23")

# ─────────────────────────────────────────────────────────────────────────────
# v24 — clean paper look: horizontal stacked, step name + ms outside as table
# ─────────────────────────────────────────────────────────────────────────────
def v24():
    fig, ax = plt.subplots(figsize=(W, 4.2))
    ys = np.arange(len(ALL_DATA))
    bh = 0.45

    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            ax.barh(ys[mi], v, left=cum, height=bh, color=pc(j),
                    edgecolor="white", linewidth=0.7)
            cum += v

    # Draw annotations as a mini table to the right
    right_x = max(TOTALS) * 1.02
    col_w   = max(TOTALS) * 0.35
    all_steps_u = list(dict.fromkeys(BASELINE_STEPS + E2E_STEPS))
    all_data_map = [dict(zip(s, v)) for _, s, v in ALL_DATA]

    for yi, dm in enumerate(all_data_map):
        for ji, step in enumerate(all_steps_u):
            v = dm.get(step, None)
            txt = f"{v:.1f}" if v else "—"
            ax.text(right_x + col_w * (ji + 0.5), ys[yi], txt,
                    ha="center", va="center", fontsize=8.5,
                    color=pc(ji) if v else "#aaa", fontweight="bold")

    # Column headers
    for ji, step in enumerate(all_steps_u):
        ax.text(right_x + col_w * (ji + 0.5), len(ALL_DATA) - 0.3,
                step, ha="center", va="bottom", fontsize=7.5,
                rotation=35, color=pc(ji), fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels(METHODS, fontsize=13)
    ax.set_xlabel("Mean time per frame (ms)", fontsize=12)
    ax.set_xlim(0, right_x + col_w * (len(all_steps_u) + 0.8))
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    save(fig, "v24")

# ─────────────────────────────────────────────────────────────────────────────
# v25 — bold typography, very minimal bars, large method total overlay
# ─────────────────────────────────────────────────────────────────────────────
def v25():
    fig, ax = plt.subplots(figsize=(W, 4.5), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    xs = np.arange(len(ALL_DATA))
    bw = 0.6

    for mi, (lbl, steps, vals) in enumerate(ALL_DATA):
        cum = 0.0
        for j, (s, v) in enumerate(zip(steps, vals)):
            ax.bar(xs[mi], v, bottom=cum, width=bw, color=pc(j),
                   edgecolor="none", linewidth=0, alpha=0.85)
            cum += v
        # Big number above
        ax.text(xs[mi], cum + 0.4, f"{cum:.1f}", ha="center", va="bottom",
                fontsize=20, fontweight="bold", color="#222")
        ax.text(xs[mi], cum + 3.2, "ms", ha="center", va="bottom",
                fontsize=10, color="#555")

    ax.set_xticks(xs)
    ax.set_xticklabels(METHODS, fontsize=14)
    ax.set_ylim(0, max(TOTALS) * 1.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(left=False, labelleft=False, bottom=False)
    ax.grid(False)

    all_steps = BASELINE_STEPS + [s for s in E2E_STEPS if s not in BASELINE_STEPS]
    handles = [mpatches.Patch(color=pc(i), label=all_steps[i], alpha=0.85)
               for i in range(len(all_steps))]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.subplots_adjust(bottom=0.18)
    save(fig, "v25")

# ─────────────────────────────────────────────────────────────────────────────
# v26 — horizontal grouped bars, one group per step, methods side by side
# ─────────────────────────────────────────────────────────────────────────────
def v26():
    all_steps = list(dict.fromkeys(BASELINE_STEPS + E2E_STEPS))
    all_data_map = [dict(zip(s, v)) for _, s, v in ALL_DATA]
    n  = len(all_steps)
    h  = 0.22
    ys = np.arange(n)
    offsets = [-h, 0, h]

    fig, ax = plt.subplots(figsize=(W, 4.5))
    for mi, (dm, color, lbl) in enumerate(zip(all_data_map, METHOD_COLORS, METHODS_LONG)):
        vals = [dm.get(s, 0) for s in all_steps]
        ax.barh(ys + offsets[mi], vals, height=h*0.85, color=color, alpha=0.9,
                edgecolor="white", linewidth=0.4, label=lbl)
        for yi, v in zip(ys + offsets[mi], vals):
            if v > 0:
                ax.text(v + 0.15, yi, f"{v:.1f}", va="center", fontsize=8.5)

    ax.set_yticks(ys)
    ax.set_yticklabels(all_steps, fontsize=10)
    ax.set_xlabel("Mean time (ms)", fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.set_xlim(0, max(VITB_VALS)*1.4)
    fig.tight_layout()
    save(fig, "v26")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Saving to {OUT_DIR}/")
    v01(); v02(); v03(); v04(); v05()
    v06(); v07(); v08(); v09(); v10()
    v11(); v12(); v13(); v14(); v15()
    v16(); v17(); v18(); v19(); v20()
    v21(); v22(); v23(); v24(); v25()
    v26()
    print("Done.")
