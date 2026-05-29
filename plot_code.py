import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# -------------------------------------------------------------------------
# 1. Style — matches graph_compare.py (plain matplotlib, no seaborn)
# -------------------------------------------------------------------------
plt.rcParams.update({
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.linewidth": 0.5,
    "grid.color": "#e0e0e0",
    "axes.grid": True,
})

colors = {
    "CPU":    "#e19c24",
    "GGRAPH": "#8fb032",
}

# -------------------------------------------------------------------------
# 2. Data (from Table 1 — B=2 and B=16 excluded as commented out)
# -------------------------------------------------------------------------
B      = np.array([1,     4,     8,    32,    64,    128,   256,   512,   1024])
ggraph = np.array([4.76,  4.85,  4.92,  6.14,  7.71, 11.61, 18.87, 33.99, 65.73])
cpu    = np.array([11.21, 53.35, 111.0, 457.0, 946.0, 1950.0, 3978.0, 8432.0, 16985.0])

# CPU measured up to B=128, extrapolated beyond (overlap at 128 connects lines).
cpu_measured_idx = B <= 128
cpu_extrap_idx   = B >= 128

# Evenly-spaced x positions (0,1,2,...) so tick spacing is equal regardless
# of the actual B values.
x = np.arange(len(B))

# -------------------------------------------------------------------------
# 3. Plot
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)

# CPU
ax.plot(x[cpu_measured_idx], cpu[cpu_measured_idx],
        color=colors["CPU"], marker="s", markersize=4,
        linewidth=1.5, label="CPU")
ax.plot(x[cpu_extrap_idx], cpu[cpu_extrap_idx],
        color=colors["CPU"], linestyle="--",
        marker="s", markerfacecolor="none", markersize=4, linewidth=1.2)

# GGRAPH — fully measured, solid throughout
ax.plot(x, ggraph,
        color=colors["GGRAPH"], marker="o", markersize=4,
        linewidth=1.5, label="GGRAPH (Ours)")

# -------------------------------------------------------------------------
# 4. Axes
# -------------------------------------------------------------------------
ax.set_yscale("log")

ax.set_ylabel("Execution Time (ms)")
ax.set_xlabel("Batch Size ($B$)")

ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in B], rotation=0, ha="center", fontsize=6.5)

ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_locator(ticker.NullLocator())

ax.tick_params(direction="in", which="both", top=True, right=True)

ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()
plt.savefig("runtime_scaling_single_column.pdf", bbox_inches="tight")
