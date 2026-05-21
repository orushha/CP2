#!/usr/bin/env python3
"""
Analysis and plotting for CP2: CPU vs GPU hash table performance.

CPU data : results/cpu/  — one CSV per thread count (JMH output)
GPU data : results/gpu/  — one CSV per GPU run (same column format)

Usage:
    python compare_hardware.py [--cpu-dir DIR] [--gpu-dir DIR] [--out-dir DIR]
"""

import argparse
import glob
import os
import re
import sys
from math import log2

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Colour palette (colour-blind-friendly) ───────────────────────────────────
IMPL_COLORS = {
    "SynchronizedMap":       "#d62728",
    "StripedMap":            "#ff7f0e",
    "StripedMapPadded":      "#2ca02c",
    "StripedWriteMap":       "#1f77b4",
    "StripedWriteMapPadded": "#9467bd",
    "StripedLevelWriteMap":  "#8c564b",
    "HashTrieMap":           "#7f7f7f",
    "WrapConcurrentHashMap": "#17becf",
    "GPUHashTable":          "#e377c2",
}

IMPL_LABELS = {
    "SynchronizedMap":       "Sync",
    "StripedMap":            "Striped",
    "StripedMapPadded":      "StripedPad",
    "StripedWriteMap":       "WriteMap",
    "StripedWriteMapPadded": "WriteMapPad",
    "StripedLevelWriteMap":  "LevelWrite",
    "HashTrieMap":           "HashTrie",
    "WrapConcurrentHashMap": "WrapCHM",
    "GPUHashTable":          "GPU (linear probing)",
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_cpu(cpu_dir: str) -> pd.DataFrame:
    """Load all JMH CSVs from cpu_dir. Thread count is parsed from filename."""
    frames = []
    for path in sorted(glob.glob(os.path.join(cpu_dir, "*.csv"))):
        m = re.search(r"-t(\d+)\.csv$", path)
        threads = int(m.group(1)) if m else 0
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.strip('"')
        df = df.rename(columns={
            "Param: distribution": "distribution",
            "Param: keyRange":     "keyRange",
            "Param: mapType":      "mapType",
            "Param: readRatio":    "readRatio",
            "Score":               "score",
            "Score Error (99.9%)": "error",
        })
        df["threads"] = threads
        df["platform"] = "CPU (DGX Spark arm64)"
        frames.append(df)
    if not frames:
        sys.exit(f"No CPU CSVs found in {cpu_dir}")
    return pd.concat(frames, ignore_index=True)


def load_gpu(gpu_dir: str) -> pd.DataFrame:
    """Load GPU result CSVs. All rows get threads=0 and platform='GPU'."""
    frames = []
    for path in sorted(glob.glob(os.path.join(gpu_dir, "*.csv"))):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.strip('"')
        df = df.rename(columns={
            "Param: distribution": "distribution",
            "Param: keyRange":     "keyRange",
            "Param: mapType":      "mapType",
            "Param: readRatio":    "readRatio",
            "Score":               "score",
            "Score Error (99.9%)": "error",
        })
        # Infer GPU name from filename (e.g., "cn11-2026-05-22_10-00-00.csv")
        basename = os.path.basename(path)
        df["platform"] = f"GPU ({basename.split('-')[0]})"
        df["threads"] = 0  # GPU has no thread count in CPU sense
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def geomean(series):
    """Geometric mean, ignoring non-positive values."""
    s = series[series > 0]
    return np.exp(np.log(s).mean()) if len(s) > 0 else float("nan")


def to_mops(val):
    return val / 1e6


# ── Figure 1: CPU scaling curves + GPU horizontal lines ─────────────────────

def fig_scaling(cpu: pd.DataFrame, gpu: pd.DataFrame, out_dir: str):
    """
    4-panel figure: rows = key range (1K / 1M), cols = read ratio (0.8 / 0.2).
    Each panel shows CPU throughput vs thread count for all implementations,
    plus a horizontal dashed line for the GPU benchmark (if available).
    Aggregated by geometric mean across distributions.
    """
    cpu_impls = [i for i in IMPL_COLORS if i != "GPUHashTable"]
    thread_counts = sorted(cpu["threads"].unique())
    key_ranges    = [1000, 1000000]
    read_ratios   = [0.8, 0.2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    fig.suptitle(
        "CPU thread scaling vs GPU throughput\n"
        "Geometric mean across distributions",
        fontsize=13
    )

    for ri, rr in enumerate(read_ratios):
        for ki, kr in enumerate(key_ranges):
            ax = axes[ki][ri]
            label_kr = "1 K" if kr == 1000 else "1 M"
            ax.set_title(f"keyRange={label_kr}  readRatio={rr}", fontsize=10)

            # CPU lines
            for impl in cpu_impls:
                sub = cpu[(cpu["mapType"] == impl) &
                          (cpu["keyRange"] == kr) &
                          (cpu["readRatio"] == rr)]
                if sub.empty:
                    continue
                pts = sub.groupby("threads")["score"].apply(geomean).reset_index()
                pts = pts.sort_values("threads")
                ax.plot(pts["threads"], pts["score"].apply(to_mops),
                        marker="o", markersize=4,
                        color=IMPL_COLORS[impl],
                        label=IMPL_LABELS[impl],
                        linewidth=1.5)

            # GPU horizontal lines
            if not gpu.empty:
                gpu_sub = gpu[(gpu["keyRange"] == kr) &
                              (gpu["readRatio"] == rr)]
                if not gpu_sub.empty:
                    for platform in gpu_sub["platform"].unique():
                        gsub = gpu_sub[gpu_sub["platform"] == platform]
                        gm = geomean(gsub["score"])
                        ax.axhline(
                            to_mops(gm),
                            color=IMPL_COLORS["GPUHashTable"],
                            linestyle="--", linewidth=2,
                            label=f"{platform} (geomean)"
                        )
                        ax.text(thread_counts[-1] * 0.65, to_mops(gm) * 1.04,
                                f"{to_mops(gm):.0f}",
                                color=IMPL_COLORS["GPUHashTable"], fontsize=8)

            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_xticks(thread_counts)
            ax.set_xlabel("CPU thread count", fontsize=9)
            ax.set_ylabel("Throughput (Mops/s)", fontsize=9)
            ax.grid(True, alpha=0.3)

    # Single legend for all panels
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(out_dir, "fig1_cpu_scaling_vs_gpu.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 2: Bar chart — CPU peak vs GPU, per configuration ─────────────────

def fig_cpu_gpu_bar(cpu: pd.DataFrame, gpu: pd.DataFrame, out_dir: str):
    """
    Bar chart comparing CPU best (WrapCHM at peak thread) with GPU
    for each of the 18 (dist × keyRange × readRatio) configurations.
    """
    if gpu.empty:
        print("Skipping fig2: no GPU data")
        return

    configs = [
        (dist, kr, rr)
        for dist in ["uniform", "zipfian_0.5", "zipfian_0.99"]
        for kr   in [1000, 1000000]
        for rr   in [0.8, 0.5, 0.2]
    ]
    peak_thread = cpu["threads"].max()

    cpu_scores, gpu_scores, labels = [], [], []
    for dist, kr, rr in configs:
        c = cpu[(cpu["threads"] == peak_thread) &
                (cpu["mapType"] == "WrapConcurrentHashMap") &
                (cpu["keyRange"] == kr) &
                (cpu["distribution"] == dist) &
                (cpu["readRatio"] == rr)]["score"]
        g = gpu[(gpu["keyRange"] == kr) &
                (gpu["distribution"] == dist) &
                (gpu["readRatio"] == rr)]["score"]
        cpu_scores.append(to_mops(c.mean()) if not c.empty else 0)
        gpu_scores.append(to_mops(g.mean()) if not g.empty else 0)
        dist_short = dist.replace("zipfian_", "z")
        labels.append(f"{dist_short}\nkr={kr//1000}K\nrr={rr}")

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(16, 5))
    bars_cpu = ax.bar(x - w/2, cpu_scores, w,
                      label=f"WrapCHM (CPU, {peak_thread}t)",
                      color=IMPL_COLORS["WrapConcurrentHashMap"])
    bars_gpu = ax.bar(x + w/2, gpu_scores, w,
                      label="GPU (linear probing)",
                      color=IMPL_COLORS["GPUHashTable"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Throughput (Mops/s)")
    ax.set_title(f"Best CPU (WrapCHM @ {peak_thread} threads) vs GPU — all 18 configurations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_cpu_peak_vs_gpu.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 3: Read-ratio sensitivity — CPU vs GPU ────────────────────────────

def fig_read_ratio(cpu: pd.DataFrame, gpu: pd.DataFrame, out_dir: str):
    """
    Line chart of throughput vs read ratio for CPU peak and GPU,
    split by key range.
    """
    peak_thread = cpu["threads"].max()
    read_ratios = sorted(cpu["readRatio"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle(
        f"Throughput vs read ratio — CPU ({peak_thread} threads) vs GPU\n"
        "Geometric mean across distributions",
        fontsize=12
    )

    for ki, kr in enumerate([1000, 1000000]):
        ax = axes[ki]
        label_kr = "1 K" if kr == 1000 else "1 M"
        ax.set_title(f"keyRange = {label_kr}", fontsize=11)

        # CPU: all implementations at peak thread count
        for impl in [i for i in IMPL_COLORS if i != "GPUHashTable"]:
            pts = []
            for rr in read_ratios:
                sub = cpu[(cpu["threads"] == peak_thread) &
                          (cpu["mapType"] == impl) &
                          (cpu["keyRange"] == kr) &
                          (cpu["readRatio"] == rr)]
                pts.append(to_mops(geomean(sub["score"])))
            ax.plot(read_ratios, pts, marker="o", markersize=5,
                    color=IMPL_COLORS[impl], label=IMPL_LABELS[impl],
                    linewidth=1.5)

        # GPU
        if not gpu.empty:
            gpu_pts = []
            for rr in read_ratios:
                gsub = gpu[(gpu["keyRange"] == kr) & (gpu["readRatio"] == rr)]
                gpu_pts.append(to_mops(geomean(gsub["score"])))
            ax.plot(read_ratios, gpu_pts, marker="D", markersize=7,
                    color=IMPL_COLORS["GPUHashTable"], linestyle="--", linewidth=2.5,
                    label=IMPL_LABELS["GPUHashTable"])

        ax.set_xlabel("Read ratio", fontsize=10)
        ax.set_ylabel("Throughput (Mops/s)", fontsize=10)
        ax.set_xticks(read_ratios)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(out_dir, "fig3_read_ratio_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 4: Distribution sensitivity — CPU vs GPU ──────────────────────────

def fig_distribution(cpu: pd.DataFrame, gpu: pd.DataFrame, out_dir: str):
    """
    Bar chart: % throughput change from uniform to zipfian_0.99 at peak thread,
    shown for CPU implementations and GPU side-by-side, split by key range.
    """
    peak_thread  = cpu["threads"].max()
    impls_cpu    = [i for i in IMPL_COLORS if i != "GPUHashTable"]
    impls_all    = impls_cpu + (["GPUHashTable"] if not gpu.empty else [])
    key_ranges   = [1000, 1000000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Throughput change: uniform → zipfian_0.99  (peak thread: CPU={peak_thread})\n"
        "Positive = skew hurts; Negative = skew helps",
        fontsize=11
    )

    for ki, kr in enumerate(key_ranges):
        ax = axes[ki]
        label_kr = "1 K" if kr == 1000 else "1 M"
        ax.set_title(f"keyRange = {label_kr}", fontsize=10)

        pcts, colors, xlabels = [], [], []
        for impl in impls_all:
            is_gpu = (impl == "GPUHashTable")
            if is_gpu:
                src = gpu
                filter_base = (src["keyRange"] == kr) & (src["distribution"] == "uniform")
                filter_skew = (src["keyRange"] == kr) & (src["distribution"] == "zipfian_0.99")
            else:
                filter_base = ((cpu["threads"] == peak_thread) &
                               (cpu["mapType"] == impl) &
                               (cpu["keyRange"] == kr) &
                               (cpu["distribution"] == "uniform"))
                filter_skew = ((cpu["threads"] == peak_thread) &
                               (cpu["mapType"] == impl) &
                               (cpu["keyRange"] == kr) &
                               (cpu["distribution"] == "zipfian_0.99"))
                src = cpu

            base = geomean(src[filter_base]["score"])
            skew = geomean(src[filter_skew]["score"])
            if base > 0 and skew > 0:
                pct = (skew - base) / base * 100
            else:
                pct = float("nan")
            pcts.append(pct)
            colors.append(IMPL_COLORS[impl])
            xlabels.append(IMPL_LABELS[impl])

        x = np.arange(len(pcts))
        bars = ax.bar(x, pcts, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, pct in zip(bars, pcts):
            if not np.isnan(pct):
                ax.text(bar.get_x() + bar.get_width()/2,
                        pct + (2 if pct >= 0 else -5),
                        f"{pct:+.0f}%", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Throughput change (%)")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_distribution_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(cpu: pd.DataFrame, gpu: pd.DataFrame):
    peak = cpu["threads"].max()
    print(f"\n=== Summary: CPU peak ({peak} threads) vs GPU (geometric mean across all configs) ===")
    print(f"{'Implementation':<28} {'CPU Mops/s':>12}  {'GPU Mops/s':>12}  {'GPU/CPU':>8}")
    print("-" * 66)

    impls = [i for i in IMPL_COLORS if i != "GPUHashTable"]
    gpu_gm = to_mops(geomean(gpu["score"])) if not gpu.empty else float("nan")

    for impl in impls:
        sub = cpu[(cpu["threads"] == peak) & (cpu["mapType"] == impl)]
        cpu_gm = to_mops(geomean(sub["score"]))
        ratio  = gpu_gm / cpu_gm if cpu_gm > 0 else float("nan")
        print(f"  {IMPL_LABELS[impl]:<26} {cpu_gm:>12.1f}  {gpu_gm:>12.1f}  {ratio:>8.2f}×")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-dir", default=os.path.join(repo, "results", "cpu"))
    ap.add_argument("--gpu-dir", default=os.path.join(repo, "results", "gpu"))
    ap.add_argument("--out-dir", default=os.path.join(repo, "results", "figures"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading CPU data from : {args.cpu_dir}")
    cpu = load_cpu(args.cpu_dir)
    print(f"  {len(cpu)} rows  threads={sorted(cpu['threads'].unique())}")

    print(f"Loading GPU data from : {args.gpu_dir}")
    gpu = load_gpu(args.gpu_dir)
    if gpu.empty:
        print("  (no GPU results yet — figures will show CPU-only where GPU is needed)")
    else:
        print(f"  {len(gpu)} rows  mapTypes={gpu['mapType'].unique().tolist()}")

    print(f"\nGenerating figures → {args.out_dir}")
    fig_scaling(cpu, gpu, args.out_dir)
    fig_cpu_gpu_bar(cpu, gpu, args.out_dir)
    fig_read_ratio(cpu, gpu, args.out_dir)
    fig_distribution(cpu, gpu, args.out_dir)

    if not gpu.empty:
        print_summary(cpu, gpu)

    print("\nDone.")


if __name__ == "__main__":
    main()
