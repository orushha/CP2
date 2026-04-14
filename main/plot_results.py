"""
plot_results.py — Visualize JMH benchmark results for concurrent HashMap study.

Handles per-thread-count CSVs (e.g. spark-c183-...-t16.csv) and single-file
layouts. Scores are auto-converted from ops/s → ops/μs.

Usage:
    python3 plot_results.py results/spark-c183-2026-03-27_19-10-10
    python3 plot_results.py results/spark-c183-2026-03-27_19-10-10 --save
"""

import sys, os, re, argparse, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

YLABEL = "Throughput (ops/μs)"

COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"
]

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

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":  True,
    "grid.alpha": 0.3,
    "font.size":  10,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def _normalise_cols(df):
    """Lowercase + underscore column names, rename param/error columns."""
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

        # threads column: prefer what JMH wrote in the CSV; fall back to filename
        if "threads" not in df.columns:
            fname = os.path.basename(path)
            m = re.search(r'-t(\d+)\.csv$', fname) or re.search(r'threads(\d+)', fname)
            df["threads"] = int(m.group(1)) if m else 1

        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"  Columns: {list(df.columns)}")

    if "ci99" not in df.columns:
        df["ci99"] = 0.0

    df["maptype"]   = df["maptype"].str.strip()
    df["score"]     = pd.to_numeric(df["score"],     errors="coerce")
    df["ci99"]      = pd.to_numeric(df["ci99"],      errors="coerce").fillna(0)
    df["threads"]   = pd.to_numeric(df["threads"],   errors="coerce").fillna(1).astype(int)
    df["keyrange"]  = pd.to_numeric(df["keyrange"],  errors="coerce").astype(int)
    df["readratio"] = pd.to_numeric(df["readratio"], errors="coerce")

    df = df[df["mode"] == "thrpt"].copy()

    # Convert ops/s → ops/μs so ylabel is consistent regardless of hardware
    if "unit" in df.columns:
        mask = df["unit"].str.strip() == "ops/s"
        df.loc[mask, "score"] /= 1e6
        df.loc[mask, "ci99"]  /= 1e6

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def filt(df, threads=None, dist=None, kr=None, ratio=None):
    d = df.copy()
    if threads is not None: d = d[d["threads"]      == threads]
    if dist    is not None: d = d[d["distribution"] == dist]
    if kr      is not None: d = d[d["keyrange"]     == kr]
    if ratio   is not None: d = d[d["readratio"]    == ratio]
    d["maptype"] = pd.Categorical(d["maptype"], categories=MAP_ORDER, ordered=True)
    return d.sort_values("maptype")

def short(m): return MAP_SHORT.get(m, m)

def save_or_show(fig, save_dir, subdir, name):
    if save_dir:
        out = os.path.join(save_dir, subdir)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, name)
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
        plt.close(fig)

def scores_errors(d, maps):
    s = [d[d["maptype"] == m]["score"].values[0] if m in d["maptype"].values else 0.0 for m in maps]
    e = [d[d["maptype"] == m]["ci99"].values[0]  if m in d["maptype"].values else 0.0 for m in maps]
    return s, e

def use_log_xaxis(thread_counts):
    """Log scale when thread range spans more than one doubling step."""
    return len(thread_counts) > 2 and max(thread_counts) / min(thread_counts) >= 8


# ── Plot 1: Peak-thread bar chart ─────────────────────────────────────────────
# Shows absolute throughput at the highest available thread count so contention
# effects and implementation differences are visible at their most extreme.

def plot_bar(df, t_snap, dist, kr, ratio, save_dir):
    d = filt(df, t_snap, dist, kr, ratio)
    if d.empty: return

    maps = [m for m in MAP_ORDER if m in d["maptype"].values]
    s, e = scores_errors(d, maps)
    if not any(v > 0 for v in s): return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([short(m) for m in maps], s, color=COLORS[:len(maps)],
                  yerr=e, capsize=4, error_kw={"linewidth": 1.2})
    ax.set_ylabel(YLABEL)
    ax.set_xlabel("Implementation")
    ax.set_title(f"Throughput at {t_snap} threads — {dist}, range {kr:,}, "
                 f"read {int(ratio*100)}%")
    fmt = ".3f" if max(s) < 1 else ".2f" if max(s) < 10 else ".1f"
    for bar, v in zip(bars, s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)
    save_or_show(fig, save_dir, "bar", f"bar_{dist}_range{kr}_r{ratio}_t{t_snap}.png")


# ── Plot 2: Scalability ───────────────────────────────────────────────────────
# Most important plot for multi-threaded analysis: throughput vs thread count.
# Uses log x-axis when thread counts span a large range (e.g. 1–64 on HPC).

def plot_scalability(df, dist, kr, ratio, save_dir):
    thread_counts = sorted(df["threads"].unique())
    if len(thread_counts) < 2:
        return

    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    fig, ax = plt.subplots(figsize=(10, 6))

    log_x = use_log_xaxis(thread_counts)

    for i, m in enumerate(maps):
        ts, s, e = [], [], []
        for t in thread_counts:
            row = filt(df, t, dist, kr, ratio)
            row = row[row["maptype"] == m]
            if not row.empty:
                ts.append(t)
                s.append(row["score"].values[0])
                e.append(row["ci99"].values[0])
        if s:
            ax.errorbar(ts, s, yerr=e, label=short(m),
                        color=COLORS[i % len(COLORS)], marker="o",
                        linewidth=1.8, capsize=3)

    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(thread_counts)

    ax.set_xlabel("Thread count")
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Scalability — {dist}, range {kr:,}, read {int(ratio*100)}%")
    ax.legend(title="Implementation", bbox_to_anchor=(1.01, 1), loc="upper left")
    save_or_show(fig, save_dir, "scalability",
                 f"scalability_{dist}_range{kr}_r{ratio}.png")


# ── Plot 3: Speedup (relative to single thread) ───────────────────────────────
# Normalises each implementation's throughput to its own t=1 baseline.
# Reveals which implementations actually scale and which saturate early.
# A flat or declining line means the implementation is contention-bound.

def plot_speedup(df, dist, kr, ratio, save_dir):
    thread_counts = sorted(df["threads"].unique())
    if len(thread_counts) < 2:
        return

    t_base = thread_counts[0]
    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    log_x = use_log_xaxis(thread_counts)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ideal speedup reference line
    ax.plot(thread_counts,
            [t / t_base for t in thread_counts],
            color="lightgray", linestyle="--", linewidth=1.2,
            label="Linear ideal", zorder=0)

    for i, m in enumerate(maps):
        base_row = filt(df, t_base, dist, kr, ratio)
        base_row = base_row[base_row["maptype"] == m]
        if base_row.empty or base_row["score"].values[0] == 0:
            continue
        base_score = base_row["score"].values[0]

        ts, speedups = [], []
        for t in thread_counts:
            row = filt(df, t, dist, kr, ratio)
            row = row[row["maptype"] == m]
            if not row.empty:
                ts.append(t)
                speedups.append(row["score"].values[0] / base_score)
        if speedups:
            ax.plot(ts, speedups, label=short(m),
                    color=COLORS[i % len(COLORS)], marker="o", linewidth=1.8)

    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(thread_counts)

    ax.set_xlabel("Thread count")
    ax.set_ylabel(f"Speedup over {t_base}T baseline")
    ax.set_title(f"Scaling efficiency — {dist}, range {kr:,}, read {int(ratio*100)}%")
    ax.legend(title="Implementation", bbox_to_anchor=(1.01, 1), loc="upper left")
    save_or_show(fig, save_dir, "speedup",
                 f"speedup_{dist}_range{kr}_r{ratio}.png")


# ── Plot 4: Distribution comparison ──────────────────────────────────────────
# Grouped bars per implementation: uniform vs zipfian_0.5 vs zipfian_0.99.
# Shows how access skew affects each implementation at peak thread count.

def plot_distribution(df, t_snap, kr, ratio, save_dir):
    dists = sorted(df["distribution"].unique())
    if len(dists) < 2: return

    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    x = np.arange(len(maps))
    w = 0.8 / len(dists)
    palette = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, dist in enumerate(dists):
        s, e = scores_errors(filt(df, t_snap, dist, kr, ratio), maps)
        offset = (i - len(dists) / 2 + 0.5) * w
        ax.bar(x + offset, s, w, label=dist,
               color=palette[i % len(palette)], yerr=e, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in maps])
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Distribution effect — range {kr:,}, read {int(ratio*100)}%, "
                 f"{t_snap}T")
    ax.legend(title="Distribution")
    save_or_show(fig, save_dir, "distribution",
                 f"dist_range{kr}_r{ratio}_t{t_snap}.png")


# ── Plot 5: Key range comparison ──────────────────────────────────────────────
# 1K vs 1M key range — shows whether cache pressure changes the rankings.

def plot_keyrange(df, t_snap, dist, ratio, save_dir):
    ranges = sorted(df["keyrange"].unique())
    if len(ranges) < 2: return

    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    x = np.arange(len(maps))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, kr in enumerate(ranges):
        s, e = scores_errors(filt(df, t_snap, dist, kr, ratio), maps)
        ax.bar(x + i * w, s, w, label=f"Range {kr:,}",
               color=["#4C72B0", "#DD8452"][i % 2], yerr=e, capsize=3)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([short(m) for m in maps])
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Key range effect — {dist}, read {int(ratio*100)}%, {t_snap}T")
    ax.legend()
    save_or_show(fig, save_dir, "keyrange",
                 f"keyrange_{dist}_r{ratio}_t{t_snap}.png")


# ── Plot 6: Read/write ratio effect ──────────────────────────────────────────
# 80/50/20% read — shows write-contention sensitivity at peak thread count.

def plot_readratio(df, t_snap, dist, kr, save_dir):
    ratios = sorted(df["readratio"].unique(), reverse=True)
    if len(ratios) < 2: return

    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    x = np.arange(len(maps))
    w = 0.8 / len(ratios)
    palette = ["#55A868", "#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ratio in enumerate(ratios):
        s, e = scores_errors(filt(df, t_snap, dist, kr, ratio), maps)
        offset = (i - len(ratios) / 2 + 0.5) * w
        label = f"{int(ratio*100)}% read"
        ax.bar(x + offset, s, w, label=label,
               color=palette[i % len(palette)], yerr=e, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in maps])
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Read/write ratio effect — {dist}, range {kr:,}, {t_snap}T")
    ax.legend(title="Workload")
    save_or_show(fig, save_dir, "readratio",
                 f"readratio_{dist}_range{kr}_t{t_snap}.png")


# ── Plot 7: Heatmap summary ───────────────────────────────────────────────────
# Implementation × distribution at peak threads. Useful overview figure.

def plot_heatmap(df, t_snap, kr, ratio, save_dir):
    dists = sorted(df["distribution"].unique())
    maps  = [m for m in MAP_ORDER if m in df["maptype"].values]
    matrix = np.zeros((len(maps), len(dists)))

    for j, dist in enumerate(dists):
        d = filt(df, t_snap, dist, kr, ratio)
        for i, m in enumerate(maps):
            row = d[d["maptype"] == m]
            if not row.empty:
                matrix[i, j] = row["score"].values[0]

    if matrix.max() == 0: return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(dists, rotation=20, ha="right")
    ax.set_yticks(range(len(maps)))
    ax.set_yticklabels([short(m) for m in maps])
    ax.set_title(f"Throughput heatmap — range {kr:,}, read {int(ratio*100)}%, "
                 f"{t_snap}T")
    plt.colorbar(im, ax=ax, label=YLABEL)
    mv = matrix.max()
    for i in range(len(maps)):
        for j in range(len(dists)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if matrix[i, j] > mv * 0.7 else "black")
    save_or_show(fig, save_dir, "heatmap",
                 f"heatmap_range{kr}_r{ratio}_t{t_snap}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot JMH concurrent HashMap benchmark results.")
    parser.add_argument("folder", help="Results folder containing CSV files")
    parser.add_argument("--save", action="store_true",
                        help="Save plots to <folder>/plots/ instead of displaying")
    args = parser.parse_args()

    save_dir = os.path.join(args.folder, "plots") if args.save else None

    print(f"Loading CSVs from {args.folder} ...")
    df = load_folder(args.folder)

    threads   = sorted(df["threads"].unique())
    dists     = sorted(df["distribution"].unique())
    keyranges = sorted(df["keyrange"].unique())
    ratios    = sorted(df["readratio"].unique())

    # Snapshot thread: use the maximum available — contention effects are most
    # visible at peak parallelism, which is what matters for HPC analysis.
    t_snap = threads[-1]
    multi  = len(threads) > 1

    print(f"  Threads:       {threads}  (snapshot at t={t_snap})")
    print(f"  Distributions: {dists}")
    print(f"  Key ranges:    {keyranges}")
    print(f"  Read ratios:   {ratios}\n")

    # ── Scalability & speedup (only meaningful with multiple thread counts) ──
    if multi:
        print("Generating scalability plots ...")
        for dist in dists:
            for kr in keyranges:
                for ratio in ratios:
                    plot_scalability(df, dist, kr, ratio, save_dir)

        print("Generating speedup plots ...")
        for dist in dists:
            for kr in keyranges:
                for ratio in ratios:
                    plot_speedup(df, dist, kr, ratio, save_dir)

    # ── Snapshot plots at t_snap ─────────────────────────────────────────────
    print(f"Generating bar charts at {t_snap}T ...")
    for dist in dists:
        for kr in keyranges:
            for ratio in ratios:
                plot_bar(df, t_snap, dist, kr, ratio, save_dir)

    print(f"Generating distribution comparison at {t_snap}T ...")
    for kr in keyranges:
        for ratio in ratios:
            plot_distribution(df, t_snap, kr, ratio, save_dir)

    print(f"Generating key range comparison at {t_snap}T ...")
    for dist in dists:
        for ratio in ratios:
            plot_keyrange(df, t_snap, dist, ratio, save_dir)

    print(f"Generating read/write ratio plots at {t_snap}T ...")
    for dist in dists:
        for kr in keyranges:
            plot_readratio(df, t_snap, dist, kr, save_dir)

    print(f"Generating heatmaps at {t_snap}T ...")
    for kr in keyranges:
        for ratio in ratios:
            plot_heatmap(df, t_snap, kr, ratio, save_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()