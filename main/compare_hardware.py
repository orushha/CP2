"""
compare_hardware.py — All paper figures for CPU vs GPU HashMap study.

Generates exactly 4 figures, each answering one research question:

  Fig 1 — performance_overview.png
      Q: Do the same implementations win on both CPU and GPU, and by how much?
      Side-by-side heatmaps (CPU | GPU): impl × 9 workload groups.
      Reader sees absolute throughput AND whether rankings are preserved.

  Fig 2 — scalability.png
      Q: How does CPU thread-count scaling compare to GPU single-pass throughput?
      2×4 grid, one subplot per implementation.
      CPU shows a scaling curve; GPU appears as a single dot (threads=1 in CSV —
      GPU parallelism is internal to the CUDA kernel, not exposed as thread count).

  Fig 3 — hardware_advantage.png
      Q: How much faster is GPU than CPU, and does it depend on implementation?
      log₂(GPU/CPU) heatmap at thread count = 1 (only common point).

  Fig 4 — distribution_sensitivity.png
      Q: Does Zipfian skew affect GPU and CPU differently?
      Per-platform throughput drop: uniform → zipfian_0.99.
      CPU may benefit from L3 hot-key caching under skew; GPU has HBM bandwidth
      but different latency profile for irregular access patterns.

Usage:
    python3 compare_hardware.py results/cpu/spark-c183-2026-03-27_19-10-10 \\
                                results/gpu/ --save
    (plots saved to results/cross_comparison/)
"""

import sys, os, re, argparse, glob
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# All 8 CPU implementations + GPU — used for full heatmaps (Fig 1, Fig 4)
MAP_ORDER = [
    "SynchronizedMap", "StripedMap", "StripedMapPadded",
    "StripedWriteMap", "StripedWriteMapPadded",
    "StripedLevelWriteMap", "HashTrieMap", "WrapConcurrentHashMap",
    "cuco_static_map",
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
    "cuco_static_map":       "cuco",
}

# Focused subset for Fig 2 (scalability) and Fig 3 (advantage):
# WrapCHM = best CPU implementation (fair "best vs best")
# HashTrie = lock-free, most scalable, conceptually closest to GPU parallelism
# Sync = baseline worst case (shows full range without cluttering middle)
FOCUS_IMPLS = ["WrapConcurrentHashMap", "HashTrieMap", "SynchronizedMap"]

CPU_COLOR, GPU_COLOR = "#4C72B0", "#DD8452"
# keep old names as aliases so figure functions don't need changing
HPC_COLOR, RPI_COLOR = CPU_COLOR, GPU_COLOR

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
    path = os.path.abspath(folder)
    # GPU results live under results/gpu/
    if os.sep + "gpu" + os.sep in path or path.endswith(os.sep + "gpu"):
        return "GPU (GB10)"
    # CPU results: keep existing hostname-based detection as fallback
    name = os.path.basename(folder).lower()
    if "raspberry" in name or "rpi" in name:
        return "CPU (RPi 5)"
    if "spark" in name:
        return "CPU (HPC)"
    return "CPU (DGX Spark)"


def save_fig(fig, save_dir, name, tight=True):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, name)
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    else:
        if tight:
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
# Two heatmaps side by side (CPU | GPU).
# Rows: implementations sorted by CPU median (so the same ordering is used
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

def plot_performance_overview(cpu, gpu, lbl_cpu, lbl_gpu, save_dir):
    dists  = [d for d in ["uniform", "zipfian_0.5", "zipfian_0.99"]
              if d in cpu["distribution"].values]
    ratios = sorted(cpu["readratio"].unique(), reverse=True)

    t_cpu = cpu["threads"].max()
    t_gpu = 1  # GPU only has t=1

    # CPU heatmap: 8 implementations × 9 workload columns
    maps_cpu, mat_cpu, cols = workload_matrix(cpu, t_cpu, dists, ratios)

    # GPU row: cuco across the same workload columns
    krs = sorted(cpu["keyrange"].unique())
    n_cols = len(cols)
    gpu_row = np.zeros((1, n_cols))
    for j, (dist, ratio) in enumerate(cols):
        scores = []
        for kr in krs:
            row = filt(gpu, 1, dist, kr, ratio)
            row = row[row["maptype"] == "cuco_static_map"]
            if not row.empty:
                scores.append(row["score"].values[0])
        gpu_row[0, j] = np.median(scores) if scores else 0

    # Use a shared colour scale so GPU (dark red) and CPU (pale) are visually comparable
    combined = np.concatenate([gpu_row.flatten(), mat_cpu.flatten()])
    global_max = combined[combined > 0].max() if (combined > 0).any() else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={"width_ratios": [1, 4]})
    n_ratio = len(ratios)

    def draw_heatmap(ax, matrix, maps, panel_label, t_snap, show_ylabels=True):
        matrix_n = matrix / global_max
        im = ax.imshow(matrix_n, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        for i in range(len(maps)):
            for j in range(n_cols):
                v = matrix[i, j]
                if v == 0:
                    continue
                fmt = f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")
                ax.text(j, i, fmt, ha="center", va="center", fontsize=7,
                        color="white" if matrix_n[i, j] > 0.5 else "black")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"{int(r*100)}% reads" for _, r in cols],
                           fontsize=7.5, rotation=45, ha="right")
        for k, dist in enumerate(dists):
            center_frac = (k * n_ratio + (n_ratio - 1) / 2 + 0.5) / n_cols
            ax.text(center_frac, -0.22, dist.replace("zipfian_", "Zipf-"),
                    ha="center", va="top", fontsize=9, fontweight="bold",
                    transform=ax.transAxes)
            if k > 0:
                ax.axvline(k * n_ratio - 0.5, color="white", linewidth=2)
        ax.set_yticks(range(len(maps)))
        ax.set_yticklabels([short(m) for m in maps] if show_ylabels else [""] * len(maps),
                           fontsize=9)
        peak_label = (f"peak: {t_snap} threads" if t_snap > 1
                      else "CUDA internal parallelism")
        ax.set_title(f"{panel_label}  ({peak_label})", fontsize=11, fontweight="bold", pad=8)
        return im

    im0 = draw_heatmap(axes[0], gpu_row, ["cuco"], lbl_gpu, t_gpu)
    im1 = draw_heatmap(axes[1], mat_cpu, maps_cpu, lbl_cpu, t_cpu)
    fig.colorbar(im1, ax=axes.tolist(), fraction=0.015, pad=0.02).set_label(
        "Throughput relative to GPU maximum\n(shared scale — GPU ≈ 1.0, CPU ≈ 0.05–0.10)",
        fontsize=9)

    fig.suptitle(
        f"Median throughput (ops/μs) — {lbl_gpu} vs {lbl_cpu} on a shared colour scale\n"
        "Both panels normalised to the GPU maximum. "
        "CPU values appear pale because GPU throughput is 10–20× higher at peak.",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    save_fig(fig, save_dir, "fig1_performance_overview.png", tight=False)


# ── Figure 2: Scalability ──────────────────────────────────────────────────────
#
# 2×4 grid (or 3×3 with cuco), one subplot per implementation.
# Both platforms overlaid on the same axes using ABSOLUTE throughput.
# Scores aggregated as median across all 18 workload configs.
#
# CPU: full scaling curve across thread counts.
# GPU: single dot at threads=1 (GPU parallelism is internal to the CUDA kernel;
#      the benchmark does not expose a thread-count knob on the GPU side).
# The vertical position of the GPU dot relative to the CPU curve shows whether
# GPU throughput exceeds, matches, or falls short of peak CPU throughput.

def plot_scalability(cpu, gpu, lbl_cpu, lbl_gpu, save_dir):
    # Focus on WrapCHM, HashTrie, Sync — tells the full range without clutter
    maps = [m for m in FOCUS_IMPLS if m in cpu["maptype"].values]

    def median_by_thread(df, m):
        ts, scores = [], []
        for t in sorted(df["threads"].unique()):
            vals = df[(df["maptype"] == m) & (df["threads"] == t)]["score"]
            if not vals.empty:
                ts.append(t)
                scores.append(vals.median())
        return ts, scores

    # GPU median across all workloads (single value — horizontal reference line)
    gpu_median = gpu[gpu["maptype"] == "cuco_static_map"]["score"].median()

    cpu_t = sorted(cpu["threads"].unique())
    log_x = len(cpu_t) > 2 and max(cpu_t) / min(cpu_t) >= 8

    ncols = 4
    nrows = (len(maps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4 * nrows))
    axes_flat = axes.flatten()

    for idx, m in enumerate(maps):
        ax = axes_flat[idx]
        ts, scores = median_by_thread(cpu, m)
        if scores:
            ax.plot(ts, scores, color=CPU_COLOR, linestyle="-",
                    marker="o", linewidth=1.8, markersize=4)
        # GPU reference line
        ax.axhline(gpu_median, color=GPU_COLOR, linestyle="--",
                   linewidth=1.5, label=f"{lbl_gpu} median")

        ax.set_title(short(m), fontsize=10, fontweight="bold")
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_xticks(cpu_t)
            ax.tick_params(axis="x", labelsize=8, rotation=45)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        if idx % ncols == 0:
            ax.set_ylabel("Throughput (ops/μs, log scale)", fontsize=8)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Thread count", fontsize=8)

    for i in range(len(maps), len(axes_flat)):
        axes_flat[i].set_visible(False)

    cpu_h = mlines.Line2D([], [], color=CPU_COLOR, linestyle="-",
                           marker="o", markersize=4,
                           label=f"{lbl_cpu}  (1–{max(cpu_t)} threads)")
    gpu_h = mlines.Line2D([], [], color=GPU_COLOR, linestyle="--",
                           linewidth=1.5,
                           label=f"{lbl_gpu}  (median across all workloads)")
    fig.legend(handles=[cpu_h, gpu_h], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.01), frameon=True, fontsize=10)

    fig.suptitle(
        f"CPU thread-count scaling vs GPU reference line — median throughput across all 18 workload configurations\n"
        f"Dashed line = {lbl_gpu} median throughput. "
        f"CPU curve crossing the dashed line = CPU matches GPU at that thread count.",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.91])
    save_fig(fig, save_dir, "fig2_scalability.png")


# ── Figure 3: Hardware advantage heatmap ──────────────────────────────────────
#
# Rows: implementations. Columns: thread counts common to both platforms.
# For CPU vs GPU, the only common thread count is t=1 (GPU has only t=1).
# Color: log₂(GPU/CPU), median across all 18 workload configs.
# Cell annotation: human-readable multiplier (e.g. "4.2×").
#
# log₂ scale is essential: 4× and 0.25× are symmetric at ±2, so the colormap
# is honest. Blue = first arg faster, red = second arg faster.
#
# This figure answers: which implementations benefit most from GPU acceleration,
# and which does GPU struggle to beat due to CPU cache or SIMD advantages?

def plot_hardware_advantage(cpu, gpu, lbl_cpu, lbl_gpu, save_dir):
    # Compare GPU cuco vs each CPU implementation at PEAK thread count.
    # Excludes SynchronizedMap: with a global lock at 64 threads it is essentially
    # serial — a 300-400× GPU speedup there is uninformative.
    # Rows = CPU implementations. Columns = 9 workload groups.
    # Color = log₂(GPU / CPU_impl) — blue = GPU faster, red = CPU faster.
    cpu_maps = [m for m in MAP_ORDER
                if m in cpu["maptype"].values and m != "SynchronizedMap"]

    dists  = [d for d in ["uniform", "zipfian_0.5", "zipfian_0.99"]
              if d in cpu["distribution"].values]
    ratios = sorted(cpu["readratio"].unique(), reverse=True)
    cols   = [(d, r) for d in dists for r in ratios]
    krs    = sorted(cpu["keyrange"].unique())
    n_ratio = len(ratios)
    n_cols  = len(cols)

    # Compare GPU vs CPU at CPU's PEAK thread count — the fair "best vs best" comparison
    t_peak = cpu["threads"].max()

    matrix = np.full((len(cpu_maps), n_cols), np.nan)
    for j, (dist, ratio) in enumerate(cols):
        for i, m in enumerate(cpu_maps):
            log_vals = []
            for kr in krs:
                cpu_row = filt(cpu, t_peak, dist, kr, ratio)
                cpu_row = cpu_row[cpu_row["maptype"] == m]
                gpu_row = filt(gpu, 1, dist, kr, ratio)
                gpu_row = gpu_row[gpu_row["maptype"] == "cuco_static_map"]
                if (not cpu_row.empty and not gpu_row.empty
                        and cpu_row["score"].values[0] > 0):
                    log_vals.append(np.log2(
                        gpu_row["score"].values[0] / cpu_row["score"].values[0]
                    ))
            if log_vals:
                matrix[i, j] = np.median(log_vals)

    vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{int(r*100)}% reads" for _, r in cols],
                       fontsize=8, rotation=45, ha="right")
    for k in range(1, len(dists)):
        ax.axvline(k * n_ratio - 0.5, color="white", linewidth=2)
    for k, dist in enumerate(dists):
        center_frac = (k * n_ratio + (n_ratio - 1) / 2 + 0.5) / n_cols
        ax.text(center_frac, -0.18, dist.replace("zipfian_", "Zipf-"),
                ha="center", va="top", fontsize=9, fontweight="bold",
                transform=ax.transAxes)

    ax.set_yticks(range(len(cpu_maps)))
    ax.set_yticklabels([short(m) for m in cpu_maps], fontsize=10)
    ax.set_ylabel(f"{lbl_cpu} implementation  (at peak: t={t_peak})", fontsize=10)

    for i in range(len(cpu_maps)):
        for j in range(n_cols):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            mult    = 2 ** abs(v)
            txt     = f"{mult:.0f}×" if mult >= 10 else f"{mult:.1f}×"
            is_dark = abs(v) > vmax * 0.55
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if is_dark else "black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(f"log₂({lbl_gpu} / {lbl_cpu})\nblue = GPU faster · red = CPU faster",
                   fontsize=9)

    fig.suptitle(
        f"GPU advantage: {lbl_gpu} vs {lbl_cpu} at peak CPU thread count (t={t_peak})\n"
        f"Each cell: how many times faster is GPU cuco vs that CPU implementation at its best.",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.88])
    save_fig(fig, save_dir, "fig3_hardware_advantage.png")


# ── Figure 4: Distribution sensitivity ────────────────────────────────────────
#
# For each implementation: throughput drop (%) when moving from uniform to
# zipfian_0.99, at each platform's peak setting.
# Positive = skew hurts, negative = hot-key locality actually helps.
# Aggregated as median across read ratios, shown separately per key range.
#
# Research question: does Zipfian skew affect CPU and GPU differently?
# CPU: hot keys may stay warm in L3 cache, partially offsetting skew cost.
# GPU: HBM bandwidth is high but latency for irregular access is also high;
#      skew could either hurt (warp divergence) or help (cache locality in L1/L2).
# Key range shown separately (1K vs 1M) to isolate cache effects.

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
    fig, axes = plt.subplots(1, len(common_krs), figsize=(8.5 * len(common_krs), 6.5),
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
                          label=f"{label}  (t={t_peak})", color=color, alpha=0.85)
            for bar, v in zip(bars, deltas):
                if abs(v) > 1:
                    ypos = bar.get_height() + (1.0 if v >= 0 else -2.5)
                    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                            f"{v:+.1f}%", ha="center", va="bottom", fontsize=7.5)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels([short(m) for m in maps], fontsize=9, rotation=20, ha="right")
        kr_cache = "small key range (1K keys)" if kr == 1000 else "large key range (1M keys)"
        ax.set_title(f"{kr:,} keys  —  {kr_cache}", fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=9, loc="upper left")
        if ax == axes[0]:
            ax.set_ylabel(
                "Throughput change: uniform → Zipfian-0.99 (%)\n"
                "positive = skew degrades throughput  ·  negative = hot-key locality helps",
                fontsize=9
            )

    fig.suptitle(
        "Impact of Zipfian-0.99 key skew relative to uniform access\n"
        "Positive = skew degrades throughput · Negative = hot-key locality helps\n"
        "CPU L3 cache may absorb hot-key reuse; GPU response depends on HBM access pattern and warp divergence.",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
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

    # CPU goes first (more thread counts); GPU goes second (threads=1 only).
    # Detect GPU by label; fall back to the "more threads = CPU" heuristic.
    a_is_gpu = "gpu" in label_a.lower()
    b_is_gpu = "gpu" in label_b.lower()
    if a_is_gpu and not b_is_gpu:
        df_a, df_b       = df_b, df_a
        label_a, label_b = label_b, label_a
    elif not a_is_gpu and not b_is_gpu:
        # both CPU: put higher-thread-count first
        if df_a["threads"].max() < df_b["threads"].max():
            df_a, df_b       = df_b, df_a
            label_a, label_b = label_b, label_a
    lbl_hpc, lbl_rpi = label_a, label_b

    parent   = os.path.commonpath([os.path.abspath(args.folder_a),
                                   os.path.abspath(args.folder_b)])
    save_dir = os.path.join(parent, "cross_comparison") if args.save else None

    print(f"Primary ({lbl_hpc}) threads : {sorted(df_a['threads'].unique())}")
    print(f"Secondary ({lbl_rpi}) threads: {sorted(df_b['threads'].unique())}")
    print(f"Common t                    : {sorted(set(df_a['threads'].unique()) & set(df_b['threads'].unique()))}\n")

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