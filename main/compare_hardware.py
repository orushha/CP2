"""
compare_hardware.py — All paper figures for cross-platform HashMap study.

Generates exactly 4 figures, each answering one research question:

  Fig 1 — performance_overview.png
      Q: Do the same implementations win on both platforms, and by how much?
      Side-by-side heatmaps (RPi | HPC): impl × 9 workload groups.
      Reader sees absolute throughput AND whether rankings are preserved.

  Fig 2 — scalability.png
      Q: Do implementations scale differently across hardware tiers?
         Where does the memory bandwidth plateau appear on each platform?
      2×4 grid, one subplot per implementation, both platforms overlaid.

  Fig 3 — hardware_advantage.png
      Q: How much faster is HPC than RPi, and does it depend on implementation
         or thread count?
      log₂(HPC/RPi) heatmap: impl × common thread counts.

  Fig 4 — distribution_sensitivity.png
      Q: Does Zipfian skew hurt more on RPi (small 2 MB L3) than on HPC,
         as predicted by the memory bandwidth and cache capacity analysis?
      Per-platform throughput drop: uniform → zipfian_0.99.

Usage:
    python3 compare_hardware.py ../results/raspberrypi-2026-03-27_18-16-22 ../results/spark-c183-2026-03-27_19-10-10 --save
    (plots saved to results/cross_comparison/)
"""

import sys, os, re, argparse, glob
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

MAP_ORDER = [
    "SynchronizedMap", "StripedMap", "StripedMapPadded",
    "StripedWriteMap", "StripedWriteMapPadded",
    "StripedLevelWriteMap", "HashTrieMap", "WrapConcurrentHashMap"
]
MAP_SHORT = {
    "SynchronizedMap":       "Sync",
    "StripedMap":            "Striped",
    "StripedMapPadded":      "StripedPad",
    "StripedWriteMap":       "WriteMap",
    "StripedWriteMapPadded": "WriteMapPad",
    "StripedLevelWriteMap":  "LevelWrite",
    "HashTrieMap":           "HashTrie",
    "WrapConcurrentHashMap": "WrapCHM",
}

HPC_COLOR, RPI_COLOR = "#4C72B0", "#DD8452"

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         10,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def _normalise_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {}
    for col in df.columns:
        if "error" in col or "99.9" in col:
            rename[col] = "ci99"
        elif col.startswith("param:_"):
            rename[col] = col[len("param:_"):]
    return df.rename(columns=rename)


def load_folder(folder):
    csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csvs:
        sys.exit(f"No CSV files found in {folder}")
    frames = []
    for path in csvs:
        df = pd.read_csv(path)
        df = _normalise_cols(df)
        if "threads" not in df.columns:
            fname = os.path.basename(path)
            m = re.search(r'-t(\d+)\.csv$', fname) or re.search(r'threads(\d+)', fname)
            df["threads"] = int(m.group(1)) if m else 1
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if "ci99" not in df.columns:
        df["ci99"] = 0.0
    df["maptype"]   = df["maptype"].str.strip()
    df["score"]     = pd.to_numeric(df["score"],     errors="coerce")
    df["ci99"]      = pd.to_numeric(df["ci99"],      errors="coerce").fillna(0)
    df["threads"]   = pd.to_numeric(df["threads"],   errors="coerce").fillna(1).astype(int)
    df["keyrange"]  = pd.to_numeric(df["keyrange"],  errors="coerce").astype(int)
    df["readratio"] = pd.to_numeric(df["readratio"], errors="coerce")
    df = df[df["mode"] == "thrpt"].copy()
    if "unit" in df.columns:
        mask = df["unit"].str.strip() == "ops/s"
        df.loc[mask, "score"] /= 1e6
        df.loc[mask, "ci99"]  /= 1e6
    return df


def detect_label(folder):
    name = os.path.basename(folder).lower()
    if "raspberry" in name or "rpi" in name:
        return "RPi 5"
    if "spark" in name:
        return f"HPC"
    return os.path.basename(folder)


def save_fig(fig, save_dir, name):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, name)
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def filt(df, threads=None, dist=None, kr=None, ratio=None):
    d = df
    if threads is not None: d = d[d["threads"]      == threads]
    if dist    is not None: d = d[d["distribution"] == dist]
    if kr      is not None: d = d[d["keyrange"]     == kr]
    if ratio   is not None: d = d[d["readratio"]    == ratio]
    return d


def short(m): return MAP_SHORT.get(m, m)


def workload_matrix(df, t_snap, dists, ratios):
    """
    Build (n_maps × n_cols) matrix where cols = (dist × ratio),
    values = median throughput across key ranges.
    Returns matrix and sorted map list (by median, descending).
    """
    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    cols = [(d, r) for d in dists for r in ratios]
    krs  = sorted(df["keyrange"].unique())

    matrix = np.zeros((len(maps), len(cols)))
    for j, (dist, ratio) in enumerate(cols):
        for i, m in enumerate(maps):
            scores = []
            for kr in krs:
                row = filt(df, t_snap, dist, kr, ratio)
                row = row[row["maptype"] == m]
                if not row.empty and row["score"].values[0] > 0:
                    scores.append(row["score"].values[0])
            matrix[i, j] = np.median(scores) if scores else 0

    order    = np.argsort(np.median(matrix, axis=1))[::-1]
    maps_s   = [maps[i] for i in order]
    matrix_s = matrix[order, :]
    return maps_s, matrix_s, cols


# ── Figure 1: Performance overview ────────────────────────────────────────────
#
# Two heatmaps side by side (RPi | HPC).
# Rows: implementations sorted by HPC median (so the same ordering is used
#       on both panels — ranking shifts immediately visible as row reorderings).
# Columns: 9 workload groups (3 distributions × 3 read ratios, median over
#          key ranges).
# Color: normalized PER COLUMN within each platform so relative rankings
#        within each workload condition are readable.
# Annotation: actual ops/μs so absolute numbers are not hidden.
#
# This single figure answers: do the same implementations win on both
# platforms? Do rankings shift under different workload conditions?
# It also shows the absolute throughput gap between platforms without
# requiring a separate figure.

def plot_performance_overview(hpc, rpi, lbl_hpc, lbl_rpi, save_dir):
    dists  = [d for d in ["uniform", "zipfian_0.5", "zipfian_0.99"]
              if d in hpc["distribution"].values and d in rpi["distribution"].values]
    ratios = sorted(
        set(hpc["readratio"].unique()) & set(rpi["readratio"].unique()),
        reverse=True
    )

    t_hpc = hpc["threads"].max()
    t_rpi = rpi["threads"].max()

    # Sort order fixed to HPC median so ranking shifts are readable
    maps_hpc, mat_hpc, cols = workload_matrix(hpc, t_hpc, dists, ratios)
    maps_rpi, mat_rpi, _    = workload_matrix(rpi, t_rpi, dists, ratios)

    # Re-order RPi rows to match HPC order
    rpi_maps_base = [m for m in MAP_ORDER if m in rpi["maptype"].values]
    rpi_order_map = {m: i for i, m in enumerate(rpi_maps_base)}
    rpi_reorder   = [rpi_order_map[m] for m in maps_hpc if m in rpi_order_map]

    # Build RPi matrix in HPC row order
    _, mat_rpi_raw, _ = workload_matrix(rpi, t_rpi, dists, ratios)
    # We need to re-map: maps_rpi sorted order → HPC order
    rpi_maps_sorted_idx = {m: i for i, m in enumerate(maps_rpi)}
    mat_rpi_reordered = np.zeros_like(mat_hpc)
    for i, m in enumerate(maps_hpc):
        if m in rpi_maps_sorted_idx:
            mat_rpi_reordered[i, :] = mat_rpi[rpi_maps_sorted_idx[m], :]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

    n_ratio = len(ratios)

    for ax, matrix, label, t_snap in [
        (axes[0], mat_rpi_reordered, lbl_rpi, t_rpi),
        (axes[1], mat_hpc,           lbl_hpc, t_hpc),
    ]:
        col_max = matrix.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1
        matrix_n = matrix / col_max

        im = ax.imshow(matrix_n, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        # Annotate with actual ops/μs
        for i in range(len(maps_hpc)):
            for j in range(len(cols)):
                v = matrix[i, j]
                if v == 0:
                    continue
                fmt = f"{v:.2f}" if v < 10 else f"{v:.0f}"
                ax.text(j, i, fmt, ha="center", va="center", fontsize=6.5,
                        color="white" if matrix_n[i, j] > 0.65 else "black")

        # X-axis: read ratio labels
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(
            [f"{int(r*100)}%" for _, r in cols], fontsize=8
        )

        # Distribution group labels above ticks
        for k, dist in enumerate(dists):
            center = k * n_ratio + (n_ratio - 1) / 2
            label_txt = dist.replace("zipfian_", "Zipf-")
            ax.text(center, -1.8, label_txt, ha="center", va="top",
                    fontsize=8.5, fontweight="bold",
                    transform=ax.get_xaxis_transform())
            if k > 0:
                ax.axvline(k * n_ratio - 0.5, color="white", linewidth=2.5)

        ax.set_yticks(range(len(maps_hpc)))
        ax.set_yticklabels(
            [short(m) for m in maps_hpc] if ax == axes[0] else [""] * len(maps_hpc),
            fontsize=9
        )
        ax.set_title(f"{label}  (t={t_snap}, peak)  ·  ops/μs",
                     fontsize=10, fontweight="bold", pad=18)

        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                     label="Relative rank\n(per-column)")

    fig.suptitle(
        "Performance overview: all implementations × workload conditions\n"
        "(rows sorted by HPC median — ranking shifts visible as row-order differences between panels)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, save_dir, "fig1_performance_overview.png")


# ── Figure 2: Scalability ──────────────────────────────────────────────────────
#
# 2×4 grid, one subplot per implementation.
# Both platforms overlaid on the same axes using ABSOLUTE throughput.
# Scores aggregated as median across all 18 workload configs.
#
# This encodes three things per subplot simultaneously:
#   • Vertical gap between lines = absolute throughput advantage of HPC
#   • Shape of each line = scaling behavior on that architecture
#   • Where each line flattens = memory bandwidth / contention saturation point
#
# The bandwidth plateau hypothesis (Chen et al. §2.2 of the paper) is directly
# testable here: RPi lines should plateau earlier and at a lower thread count.

def plot_scalability(hpc, rpi, lbl_hpc, lbl_rpi, save_dir):
    maps = [m for m in MAP_ORDER
            if m in hpc["maptype"].values or m in rpi["maptype"].values]

    def median_by_thread(df, m):
        ts, scores = [], []
        for t in sorted(df["threads"].unique()):
            vals = df[(df["maptype"] == m) & (df["threads"] == t)]["score"]
            if not vals.empty:
                ts.append(t)
                scores.append(vals.median())
        return ts, scores

    all_t = sorted(set(hpc["threads"].unique()) | set(rpi["threads"].unique()))
    log_x = len(all_t) > 2 and max(all_t) / min(all_t) >= 8

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes_flat = axes.flatten()

    for idx, m in enumerate(maps):
        ax = axes_flat[idx]
        for df, color, label, ls in [
            (hpc, HPC_COLOR, lbl_hpc, "-"),
            (rpi, RPI_COLOR, lbl_rpi, "--"),
        ]:
            ts, scores = median_by_thread(df, m)
            if scores:
                ax.plot(ts, scores, label=label, color=color,
                        linestyle=ls, marker="o", linewidth=1.8, markersize=4)

        ax.set_title(short(m), fontsize=10, fontweight="bold")
        if log_x:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_xticks(all_t)
            ax.tick_params(axis="x", labelsize=7, rotation=45)
        if idx % 4 == 0:
            ax.set_ylabel("ops/μs", fontsize=8)
        if idx >= 4:
            ax.set_xlabel("Threads", fontsize=8)

    for i in range(len(maps), len(axes_flat)):
        axes_flat[i].set_visible(False)

    hpc_h = mlines.Line2D([], [], color=HPC_COLOR, linestyle="-",
                           marker="o", markersize=4, label=lbl_hpc)
    rpi_h = mlines.Line2D([], [], color=RPI_COLOR, linestyle="--",
                           marker="o", markersize=4, label=lbl_rpi)
    fig.legend(handles=[hpc_h, rpi_h], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), frameon=True, fontsize=10)

    fig.suptitle(
        "Throughput scaling: both platforms\n"
        "(absolute ops/μs · median across all 18 workload configs · "
        "flattening = bandwidth/contention saturation)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    save_fig(fig, save_dir, "fig2_scalability.png")


# ── Figure 3: Hardware advantage heatmap ──────────────────────────────────────
#
# Rows: implementations. Columns: thread counts common to both platforms.
# Color: log₂(HPC/RPi), median across all 18 workload configs.
# Cell annotation: human-readable multiplier (e.g. "4.2×").
#
# log₂ scale is essential: 4× and 0.25× are symmetric at ±2, so the colormap
# is honest. Blue = HPC faster, red = RPi faster (unexpected — would suggest
# the RPi's simpler memory hierarchy benefits that implementation).
#
# This figure answers: is the hardware gap uniform across implementations and
# thread counts, or do some implementations close the gap at high thread counts
# (suggesting contention dominates over raw compute on the HPC side)?

def plot_hardware_advantage(hpc, rpi, lbl_hpc, lbl_rpi, save_dir):
    common_t = sorted(
        set(hpc["threads"].unique()) & set(rpi["threads"].unique())
    )
    if not common_t:
        print("  Skipping hardware advantage: no common thread counts.")
        return

    maps = [m for m in MAP_ORDER
            if m in hpc["maptype"].values and m in rpi["maptype"].values]
    common_dists  = set(hpc["distribution"].unique()) & set(rpi["distribution"].unique())
    common_krs    = set(hpc["keyrange"].unique())     & set(rpi["keyrange"].unique())
    common_ratios = set(hpc["readratio"].unique())    & set(rpi["readratio"].unique())

    matrix = np.full((len(maps), len(common_t)), np.nan)
    for j, t in enumerate(common_t):
        for i, m in enumerate(maps):
            log_ratios = []
            for dist in common_dists:
                for kr in common_krs:
                    for ratio in common_ratios:
                        a = filt(hpc, t, dist, kr, ratio)
                        a = a[a["maptype"] == m]
                        b = filt(rpi, t, dist, kr, ratio)
                        b = b[b["maptype"] == m]
                        if (not a.empty and not b.empty
                                and b["score"].values[0] > 0
                                and a["score"].values[0] > 0):
                            log_ratios.append(
                                np.log2(a["score"].values[0] /
                                        b["score"].values[0])
                            )
            if log_ratios:
                matrix[i, j] = np.median(log_ratios)

    vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1

    fig, ax = plt.subplots(figsize=(max(6, len(common_t) * 1.8 + 2), 6))
    im = ax.imshow(matrix, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(common_t)))
    ax.set_xticklabels([str(t) for t in common_t])
    ax.set_yticks(range(len(maps)))
    ax.set_yticklabels([short(m) for m in maps])
    ax.set_xlabel("Thread count (common to both platforms)")

    for i in range(len(maps)):
        for j in range(len(common_t)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            mult    = 2 ** abs(v)
            txt     = f"{mult:.1f}×" if abs(v) > 0.15 else "≈1×"
            is_dark = abs(v) > vmax * 0.55
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if is_dark else "black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(f"log₂ ({lbl_hpc} / {lbl_rpi})  [0 = equal]")
    ax.set_title(
        f"Hardware throughput advantage: {lbl_hpc} vs {lbl_rpi}\n"
        f"(blue = HPC faster · red = RPi faster · "
        f"median across all 18 workload configs)",
        fontsize=10
    )
    plt.tight_layout()
    save_fig(fig, save_dir, "fig3_hardware_advantage.png")


# ── Figure 4: Distribution sensitivity ────────────────────────────────────────
#
# Tests the Zipfian hypothesis from §2.2 of the paper:
#   "With a highly-skewed Zipfian distribution... the degradation on RPi is
#    likely to be more significant than Chen et al. observed."
#
# For each implementation: throughput drop (%) when moving from uniform to
# zipfian_0.99, at each platform's peak thread count.
# Positive = skew hurts, negative = cache locality from hot keys actually helps.
# Aggregated as median across key ranges and read ratios.
#
# If RPi bars are consistently taller than HPC bars: hypothesis confirmed —
# the 2MB L3 is too small to hold hot keys, adding cache misses on top of
# synchronization costs exactly as predicted.
# Key range is shown separately (1K vs 1M) because the effect of the RPi's
# small L3 is most visible at 1M keys where the working set cannot fit.

def plot_distribution_sensitivity(hpc, rpi, lbl_hpc, lbl_rpi, save_dir):
    for df in [hpc, rpi]:
        if ("uniform" not in df["distribution"].values or
                "zipfian_0.99" not in df["distribution"].values):
            print("  Skipping distribution sensitivity: missing required distributions.")
            return

    maps = [m for m in MAP_ORDER
            if m in hpc["maptype"].values or m in rpi["maptype"].values]

    common_krs    = sorted(set(hpc["keyrange"].unique()) & set(rpi["keyrange"].unique()))
    common_ratios = sorted(set(hpc["readratio"].unique()) & set(rpi["readratio"].unique()))

    # Two subplots: one per key range
    # The L3 hypothesis is most testable at 1M keys (working set exceeds RPi L3)
    fig, axes = plt.subplots(1, len(common_krs), figsize=(7 * len(common_krs), 5),
                             sharey=True)
    if len(common_krs) == 1:
        axes = [axes]

    x = np.arange(len(maps))
    w = 0.35

    for ax, kr in zip(axes, common_krs):
        for k, (df, color, label) in enumerate([
            (hpc, HPC_COLOR, lbl_hpc),
            (rpi, RPI_COLOR, lbl_rpi),
        ]):
            t_peak = df["threads"].max()
            deltas = []
            for m in maps:
                vals = []
                for ratio in common_ratios:
                    uni = filt(df, t_peak, "uniform",      kr, ratio)
                    uni = uni[uni["maptype"] == m]
                    skw = filt(df, t_peak, "zipfian_0.99", kr, ratio)
                    skw = skw[skw["maptype"] == m]
                    if (not uni.empty and not skw.empty
                            and uni["score"].values[0] > 0):
                        pct = ((uni["score"].values[0] - skw["score"].values[0])
                               / uni["score"].values[0]) * 100
                        vals.append(pct)
                deltas.append(np.median(vals) if vals else 0)

            bars = ax.bar(x + k * w, deltas, w,
                          label=f"{label} (t={t_peak})", color=color, alpha=0.85)
            for bar, v in zip(bars, deltas):
                if abs(v) > 1:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.5 if v >= 0 else -1.5),
                            f"{v:+.1f}%", ha="center", va="bottom", fontsize=7)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels([short(m) for m in maps], fontsize=8)
        ax.set_title(f"Key range: {kr:,} keys", fontsize=10)
        ax.legend(fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Throughput drop: uniform → zipfian_0.99 (%)\n"
                          "positive = skew hurts · negative = hot-key cache locality helps")

    fig.suptitle(
        "Distribution sensitivity by platform at peak thread count\n"
        "(tests hypothesis: Zipfian degradation is worse on RPi's 2 MB L3 than on HPC · "
        "median across read ratios)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, save_dir, "fig4_distribution_sensitivity.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_a", help="First results folder")
    parser.add_argument("folder_b", help="Second results folder")
    parser.add_argument("--save", action="store_true",
                        help="Save to results/cross_comparison/")
    args = parser.parse_args()

    label_a = detect_label(args.folder_a)
    label_b = detect_label(args.folder_b)

    print(f"Loading {label_a}  ← {args.folder_a}")
    df_a = load_folder(args.folder_a)
    print(f"Loading {label_b}  ← {args.folder_b}\n")
    df_b = load_folder(args.folder_b)

    # HPC = platform with more thread counts
    if df_a["threads"].max() < df_b["threads"].max():
        df_a, df_b     = df_b, df_a
        label_a, label_b = label_b, label_a
    lbl_hpc, lbl_rpi = label_a, label_b

    parent   = os.path.commonpath([os.path.abspath(args.folder_a),
                                   os.path.abspath(args.folder_b)])
    save_dir = os.path.join(parent, "cross_comparison") if args.save else None

    print(f"HPC threads : {sorted(df_a['threads'].unique())}")
    print(f"RPi threads : {sorted(df_b['threads'].unique())}")
    print(f"Common t    : {sorted(set(df_a['threads'].unique()) & set(df_b['threads'].unique()))}\n")

    print("Figure 1: Performance overview ...")
    plot_performance_overview(df_a, df_b, lbl_hpc, lbl_rpi, save_dir)

    print("Figure 2: Scalability ...")
    plot_scalability(df_a, df_b, lbl_hpc, lbl_rpi, save_dir)

    print("Figure 3: Hardware advantage heatmap ...")
    plot_hardware_advantage(df_a, df_b, lbl_hpc, lbl_rpi, save_dir)

    print("Figure 4: Distribution sensitivity ...")
    plot_distribution_sensitivity(df_a, df_b, lbl_hpc, lbl_rpi, save_dir)

    print("\nDone! 4 figures generated.")

if __name__ == "__main__":
    main()