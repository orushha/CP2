"""
compare_hardware.py — Cross-platform comparison of concurrent HashMap benchmarks.

Produces 4 focused comparative plots:
  1. Scaling efficiency overlay  — speedup curves for both platforms, one subplot
                                   per implementation (2×4 grid)
  2. Hardware advantage heatmap  — log₂(HPC/RPi) throughput ratio at each common
                                   thread count, median across all workload configs
  3. Per-core efficiency         — throughput/thread at peak parallelism, median
                                   across all workload configs
  4. Distribution sensitivity    — how much uniform→zipfian_0.99 hurts each
                                   implementation on each platform

The platform with more thread counts is automatically treated as "HPC".

Usage:
    python3 compare_hardware.py results/raspberrypi-2026-03-27_18-16-22 \\
                                results/spark-c183-2026-03-27_19-10-10
    python3 compare_hardware.py <folder_a> <folder_b> --save
"""

import sys, os, re, argparse, glob
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

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

# HPC = solid blue, RPi = dashed orange — consistent across all 4 plots
HPC_COLOR, RPI_COLOR = "#4C72B0", "#DD8452"
HPC_LS,    RPI_LS    = "-",       "--"

plt.rcParams.update({
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         10,
})


# ── Data loading (mirrors plot_results.py) ─────────────────────────────────────

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
        return "RPi 5 (aarch64)"
    if "spark" in name:
        return f"HPC ({os.path.basename(folder).split('-')[0]})"
    return os.path.basename(folder)


def save_fig(fig, save_dir, name):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, name)
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    else:
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


# ── Plot 1: Scaling efficiency overlay ────────────────────────────────────────
#
# 2×4 grid — one subplot per implementation.
# Both platforms' speedup curves (normalised to their own t=1) are overlaid on
# the same axes. Reveals which implementations scale differently per architecture
# regardless of absolute throughput differences between the machines.

def plot_scalability_overlay(hpc, rpi, lbl_hpc, lbl_rpi, dist, kr, ratio, save_dir):
    maps = [m for m in MAP_ORDER
            if m in hpc["maptype"].values or m in rpi["maptype"].values]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes_flat = axes.flatten()

    all_threads = sorted(set(hpc["threads"].unique()) | set(rpi["threads"].unique()))

    for idx, m in enumerate(maps):
        ax = axes_flat[idx]

        # Light ideal reference spanning the full thread range
        ax.plot(all_threads, [t / all_threads[0] for t in all_threads],
                color="#d0d0d0", linestyle=":", linewidth=1.1, zorder=0)

        for df, color, ls, label in [
            (hpc, HPC_COLOR, HPC_LS, lbl_hpc),
            (rpi, RPI_COLOR, RPI_LS, lbl_rpi),
        ]:
            ts_avail = sorted(df["threads"].unique())
            base_row = filt(df, ts_avail[0], dist, kr, ratio)
            base_row = base_row[base_row["maptype"] == m]
            if base_row.empty or base_row["score"].values[0] == 0:
                continue
            base = base_row["score"].values[0]

            pts = [(t, filt(df, t, dist, kr, ratio))  for t in ts_avail]
            pts = [(t, d[d["maptype"] == m]) for t, d in pts if not d[d["maptype"] == m].empty]
            if not pts:
                continue

            ts_plot  = [t for t, _ in pts]
            speedups = [d["score"].values[0] / base for _, d in pts]

            ax.plot(ts_plot, speedups, label=label, color=color, linestyle=ls,
                    marker="o", linewidth=1.8, markersize=4)

        ax.set_title(short(m), fontsize=10, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(all_threads)
        ax.tick_params(axis="x", labelsize=7, rotation=45)

        if idx % 4 == 0:
            ax.set_ylabel("Speedup (×)", fontsize=9)
        if idx >= 4:
            ax.set_xlabel("Threads", fontsize=9)

    # Remove unused subplots
    for i in range(len(maps), len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Figure-level legend
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    ideal_h = mlines.Line2D([], [], color="#d0d0d0", linestyle=":", linewidth=1.1)
    fig.legend(handles + [ideal_h], lbls + ["Linear ideal"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(
        f"Scaling efficiency — {dist}, range {kr:,}, read {int(ratio*100)}%\n"
        f"(speedup normalised to each platform's own single-thread baseline)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    save_fig(fig, save_dir, f"compare_scalability_{dist}_range{kr}_r{ratio}.png")


# ── Plot 2: Hardware advantage heatmap ────────────────────────────────────────
#
# At each common thread count, compute log₂(HPC / RPi) throughput.
# Aggregated as the median across ALL workload configurations.
# Blue = HPC faster, red = RPi faster. Cell annotations show the raw multiplier.
# Immediately reveals which implementations are architecture-sensitive.

def plot_hardware_advantage(hpc, rpi, lbl_hpc, lbl_rpi, save_dir):
    common_t = sorted(set(hpc["threads"].unique()) & set(rpi["threads"].unique()))
    maps = [m for m in MAP_ORDER
            if m in hpc["maptype"].values and m in rpi["maptype"].values]

    matrix = np.full((len(maps), len(common_t)), np.nan)

    dists  = list(set(hpc["distribution"].unique()) & set(rpi["distribution"].unique()))
    krs    = list(set(hpc["keyrange"].unique())     & set(rpi["keyrange"].unique()))
    ratios = list(set(hpc["readratio"].unique())    & set(rpi["readratio"].unique()))

    for j, t in enumerate(common_t):
        for i, m in enumerate(maps):
            log_ratios = []
            for dist in dists:
                for kr in krs:
                    for ratio in ratios:
                        a = filt(hpc, t, dist, kr, ratio)
                        a = a[a["maptype"] == m]
                        b = filt(rpi, t, dist, kr, ratio)
                        b = b[b["maptype"] == m]
                        if (not a.empty and not b.empty
                                and b["score"].values[0] > 0
                                and a["score"].values[0] > 0):
                            log_ratios.append(
                                np.log2(a["score"].values[0] / b["score"].values[0])
                            )
            if log_ratios:
                matrix[i, j] = np.median(log_ratios)

    vmax = np.nanmax(np.abs(matrix))

    fig, ax = plt.subplots(figsize=(max(6, len(common_t) * 1.5 + 2), 6))
    im = ax.imshow(matrix, cmap="RdBu", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(common_t)))
    ax.set_xticklabels([str(t) for t in common_t])
    ax.set_yticks(range(len(maps)))
    ax.set_yticklabels([short(m) for m in maps])
    ax.set_xlabel("Thread count (common to both platforms)")

    ax.set_title(
        f"Hardware advantage: log₂({lbl_hpc} / {lbl_rpi})\n"
        f"(median across all workload configs · blue = HPC faster · red = RPi faster)",
        fontsize=10
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("log₂ ratio  (0 = equal)")

    for i in range(len(maps)):
        for j in range(len(common_t)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            mult = 2 ** abs(v)
            txt  = f"{mult:.1f}×" if abs(v) > 0.15 else "≈1×"
            dark = abs(v) > vmax * 0.55
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if dark else "black")

    plt.tight_layout()
    save_fig(fig, save_dir, "compare_hardware_advantage.png")


# ── Plot 3: Per-core efficiency ────────────────────────────────────────────────
#
# Throughput / thread_count at each platform's peak thread count.
# Median across all workload configs.
# A high bar means the implementation extracts useful work from each core;
# a low bar means extra threads add contention rather than throughput.

def plot_percore_efficiency(hpc, rpi, lbl_hpc, lbl_rpi, save_dir):
    maps = [m for m in MAP_ORDER
            if m in hpc["maptype"].values or m in rpi["maptype"].values]
    x = np.arange(len(maps))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (df, label, color) in enumerate([
        (hpc, lbl_hpc, HPC_COLOR),
        (rpi, lbl_rpi, RPI_COLOR),
    ]):
        t_peak = df["threads"].max()
        medians = []
        for m in maps:
            vals = []
            for dist in df["distribution"].unique():
                for kr in df["keyrange"].unique():
                    for ratio in df["readratio"].unique():
                        row = filt(df, t_peak, dist, kr, ratio)
                        row = row[row["maptype"] == m]
                        if not row.empty and row["score"].values[0] > 0:
                            vals.append(row["score"].values[0] / t_peak)
            medians.append(np.median(vals) if vals else 0)

        bars = ax.bar(x + i * w, medians, w, label=f"{label} (t={t_peak})", color=color)

        # Annotate top of each bar
        for bar, v in zip(bars, medians):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([short(m) for m in maps])
    ax.set_ylabel("ops/μs per thread")
    ax.set_title(
        "Per-core efficiency at peak parallelism\n"
        "(median across all workload configs — higher = scales better per core)"
    )
    ax.legend()
    plt.tight_layout()
    save_fig(fig, save_dir, "compare_percore_efficiency.png")


# ── Plot 4: Distribution sensitivity by platform ───────────────────────────────
#
# For each (platform, implementation): how much does throughput change when
# switching from uniform to zipfian_0.99 (highly skewed hot-key access)?
# Metric: (score_uniform − score_zip99) / score_uniform, aggregated as median
# across key ranges × read ratios, at each platform's peak thread count.
# Positive = skew hurts, negative = skew actually helps (cache locality effect).

def plot_distribution_sensitivity(hpc, rpi, lbl_hpc, lbl_rpi, save_dir):
    maps = [m for m in MAP_ORDER
            if m in hpc["maptype"].values or m in rpi["maptype"].values]
    x = np.arange(len(maps))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (df, label, color) in enumerate([
        (hpc, lbl_hpc, HPC_COLOR),
        (rpi, lbl_rpi, RPI_COLOR),
    ]):
        if ("uniform"      not in df["distribution"].values or
                "zipfian_0.99" not in df["distribution"].values):
            print(f"  Skipping {label}: missing uniform or zipfian_0.99")
            continue

        t_peak = df["threads"].max()
        sensitivities = []

        for m in maps:
            deltas = []
            for kr in df["keyrange"].unique():
                for ratio in df["readratio"].unique():
                    uni = filt(df, t_peak, "uniform",      kr, ratio)
                    uni = uni[uni["maptype"] == m]
                    skw = filt(df, t_peak, "zipfian_0.99", kr, ratio)
                    skw = skw[skw["maptype"] == m]
                    if (not uni.empty and not skw.empty
                            and uni["score"].values[0] > 0):
                        delta = ((uni["score"].values[0] - skw["score"].values[0])
                                 / uni["score"].values[0])
                        deltas.append(delta)
            sensitivities.append(np.median(deltas) * 100 if deltas else 0)

        bars = ax.bar(x + i * w, sensitivities, w,
                      label=f"{label} (t={t_peak})", color=color)

        for bar, v in zip(bars, sensitivities):
            if abs(v) > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (1 if v >= 0 else -2),
                        f"{v:+.1f}%", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8, zorder=3)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([short(m) for m in maps])
    ax.set_ylabel("Throughput drop uniform → zipfian_0.99 (%)")
    ax.set_title(
        "Distribution sensitivity by platform at peak thread count\n"
        "(positive = hot-key skew hurts · negative = skew helps via cache locality · "
        "median across key ranges and read ratios)"
    )
    ax.legend()
    plt.tight_layout()
    save_fig(fig, save_dir, "compare_distribution_sensitivity.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform comparison of HashMap benchmark results.")
    parser.add_argument("folder_a", help="First results folder")
    parser.add_argument("folder_b", help="Second results folder")
    parser.add_argument("--save", action="store_true",
                        help="Save to results/cross_comparison/ instead of displaying")
    args = parser.parse_args()

    label_a = detect_label(args.folder_a)
    label_b = detect_label(args.folder_b)

    print(f"Loading {label_a}  ← {args.folder_a}")
    df_a = load_folder(args.folder_a)
    print(f"Loading {label_b}  ← {args.folder_b}\n")
    df_b = load_folder(args.folder_b)

    # Always put the platform with more thread counts (HPC) first
    if df_a["threads"].max() < df_b["threads"].max():
        df_a, df_b     = df_b, df_a
        label_a, label_b = label_b, label_a

    lbl_hpc, lbl_rpi = label_a, label_b

    # Save into a dedicated cross-comparison folder alongside the result folders
    parent   = os.path.commonpath([os.path.abspath(args.folder_a),
                                   os.path.abspath(args.folder_b)])
    save_dir = os.path.join(parent, "cross_comparison") if args.save else None

    # Intersection of workload configs present in both datasets
    common_dists  = sorted(set(df_a["distribution"].unique()) & set(df_b["distribution"].unique()))
    common_krs    = sorted(set(df_a["keyrange"].unique())     & set(df_b["keyrange"].unique()))
    common_ratios = sorted(set(df_a["readratio"].unique())    & set(df_b["readratio"].unique()))

    print(f"Common distributions : {common_dists}")
    print(f"Common key ranges    : {common_krs}")
    print(f"Common read ratios   : {common_ratios}")
    print(f"HPC threads          : {sorted(df_a['threads'].unique())}")
    print(f"RPi threads          : {sorted(df_b['threads'].unique())}\n")

    # ── Plot 1: Scalability overlay ──────────────────────────────────────────
    # Generate for uniform @ 80% read for each key range — the clearest story.
    print("Generating scalability overlay plots ...")
    if "uniform" in common_dists and 0.8 in common_ratios:
        for kr in common_krs:
            plot_scalability_overlay(
                df_a, df_b, lbl_hpc, lbl_rpi,
                "uniform", kr, 0.8, save_dir
            )

    # ── Plot 2: Hardware advantage heatmap ───────────────────────────────────
    print("Generating hardware advantage heatmap ...")
    plot_hardware_advantage(df_a, df_b, lbl_hpc, lbl_rpi, save_dir)

    # ── Plot 3: Per-core efficiency ──────────────────────────────────────────
    print("Generating per-core efficiency plot ...")
    plot_percore_efficiency(df_a, df_b, lbl_hpc, lbl_rpi, save_dir)

    # ── Plot 4: Distribution sensitivity ────────────────────────────────────
    print("Generating distribution sensitivity plot ...")
    plot_distribution_sensitivity(df_a, df_b, lbl_hpc, lbl_rpi, save_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()