"""
plot_results.py — Visualize JMH benchmark results for concurrent HashMap study.

Usage:
    python3 plot_results.py results/raspberrypi-22-03-2026_06-21-07
    python3 plot_results.py results/raspberrypi-22-03-2026_06-21-07 --save
"""

import sys, os, argparse, glob
import pandas as pd
import matplotlib.pyplot as plt
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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_folder(folder):
    csvs = glob.glob(os.path.join(folder, "*.csv"))
    if not csvs:
        sys.exit(f"No CSV files found in {folder}")

    frames = []
    for path in sorted(csvs):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        fname = os.path.basename(path)
        if "threads" in fname:
            try:
                t = int(''.join(filter(str.isdigit,
                    fname.split("threads")[1].split(".")[0])))
                df["threads"] = t
            except Exception:
                pass
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"  Columns found: {list(df.columns)}")

    # Rename error and param columns
    rename = {}
    for col in df.columns:
        if "error" in col or "99.9" in col:
            rename[col] = "ci99"
        elif col.startswith("param:_"):
            rename[col] = col.replace("param:_", "")
    df = df.rename(columns=rename)

    if "ci99" not in df.columns:
        df["ci99"] = 0.0

    df["maptype"]   = df["maptype"].str.strip()
    df["score"]     = pd.to_numeric(df["score"],     errors="coerce")
    df["ci99"]      = pd.to_numeric(df["ci99"],      errors="coerce").fillna(0)
    df["threads"]   = pd.to_numeric(df["threads"],   errors="coerce").fillna(1).astype(int)
    df["keyrange"]  = pd.to_numeric(df["keyrange"],  errors="coerce").astype(int)
    df["readratio"] = pd.to_numeric(df["readratio"], errors="coerce")

    # Keep only throughput rows
    df = df[df["mode"] == "thrpt"].copy()
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
    s = [d[d["maptype"] == m]["score"].values[0] if m in d["maptype"].values else 0 for m in maps]
    e = [d[d["maptype"] == m]["ci99"].values[0]  if m in d["maptype"].values else 0 for m in maps]
    return s, e


# ── Plot 1: Bar chart ─────────────────────────────────────────────────────────

def plot_bar(df, threads, dist, kr, ratio, save_dir):
    d = filt(df, threads, dist, kr, ratio)
    if d.empty: return

    maps = [m for m in MAP_ORDER if m in d["maptype"].values]
    s, e = scores_errors(d, maps)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([short(m) for m in maps], s, color=COLORS[:len(maps)],
                  yerr=e, capsize=4, error_kw={"linewidth": 1.2})
    ax.set_ylabel(YLABEL)
    ax.set_xlabel("Implementation")
    ax.set_title(f"Throughput — {dist}, range {kr:,}, read {int(ratio*100)}%, {threads}T")
    fmt = ".2f" if max(s) < 5 else ".1f"
    for bar, v in zip(bars, s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)
    save_or_show(fig, save_dir, "bar", f"bar_{dist}_range{kr}_r{ratio}_t{threads}.png")


# ── Plot 2: Key range comparison ──────────────────────────────────────────────

def plot_keyrange(df, threads, dist, ratio, save_dir):
    ranges = sorted(df["keyrange"].unique())
    if len(ranges) < 2: return

    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    x, w = np.arange(len(maps)), 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, kr in enumerate(ranges):
        s, e = scores_errors(filt(df, threads, dist, kr, ratio), maps)
        ax.bar(x + i*w, s, w, label=f"Range {kr:,}",
               color=["#4C72B0","#DD8452"][i], yerr=e, capsize=3)
    ax.set_xticks(x + w/2)
    ax.set_xticklabels([short(m) for m in maps])
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Key range effect — {dist}, read {int(ratio*100)}%, {threads}T")
    ax.legend()
    save_or_show(fig, save_dir, "keyrange", f"keyrange_{dist}_r{ratio}_t{threads}.png")


# ── Plot 3: Distribution comparison ──────────────────────────────────────────

def plot_distribution(df, threads, kr, ratio, save_dir):
    dists = sorted(df["distribution"].unique())
    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    x, w = np.arange(len(maps)), 0.8 / len(dists)
    palette = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, dist in enumerate(dists):
        s, e = scores_errors(filt(df, threads, dist, kr, ratio), maps)
        offset = (i - len(dists)/2 + 0.5) * w
        ax.bar(x + offset, s, w, label=dist, color=palette[i % 3], yerr=e, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in maps])
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Distribution effect — range {kr:,}, read {int(ratio*100)}%, {threads}T")
    ax.legend(title="Distribution")
    save_or_show(fig, save_dir, "distribution", f"dist_range{kr}_r{ratio}_t{threads}.png")


# ── Plot 4: Read/write ratio effect ──────────────────────────────────────────

def plot_readratio(df, threads, dist, kr, save_dir):
    ratios = sorted(df["readratio"].unique(), reverse=True)
    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    x, w = np.arange(len(maps)), 0.8 / len(ratios)
    palette = ["#55A868", "#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ratio in enumerate(ratios):
        s, e = scores_errors(filt(df, threads, dist, kr, ratio), maps)
        offset = (i - len(ratios)/2 + 0.5) * w
        label = f"{int(ratio*100)}% read / {int((1-ratio)*100)}% write"
        ax.bar(x + offset, s, w, label=label, color=palette[i % 3], yerr=e, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in maps])
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Read/write ratio effect — {dist}, range {kr:,}, {threads}T")
    ax.legend(title="Workload")
    save_or_show(fig, save_dir, "readratio", f"readratio_{dist}_range{kr}_t{threads}.png")


# ── Plot 5: Scalability ───────────────────────────────────────────────────────

def plot_scalability(df, dist, kr, ratio, save_dir):
    thread_counts = sorted(df["threads"].unique())
    if len(thread_counts) < 2:
        return

    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(maps):
        s, e, ts = [], [], []
        for t in thread_counts:
            row = filt(df, t, dist, kr, ratio)
            row = row[row["maptype"] == m]
            if not row.empty:
                s.append(row["score"].values[0])
                e.append(row["ci99"].values[0])
                ts.append(t)
        if s:
            ax.errorbar(ts, s, yerr=e, label=short(m),
                        color=COLORS[i % len(COLORS)], marker="o",
                        linewidth=1.8, capsize=3)
    ax.set_xlabel("Thread count")
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Scalability — {dist}, range {kr:,}, read {int(ratio*100)}%")
    ax.set_xticks(thread_counts)
    ax.legend(title="Implementation", bbox_to_anchor=(1.01, 1), loc="upper left")
    save_or_show(fig, save_dir, "scalability", f"scalability_{dist}_range{kr}_r{ratio}.png")


# ── Plot 6: Heatmap ───────────────────────────────────────────────────────────

def plot_heatmap(df, threads, kr, ratio, save_dir):
    dists = sorted(df["distribution"].unique())
    maps = [m for m in MAP_ORDER if m in df["maptype"].values]
    matrix = np.zeros((len(maps), len(dists)))
    for j, dist in enumerate(dists):
        d = filt(df, threads, dist, kr, ratio)
        for i, m in enumerate(maps):
            row = d[d["maptype"] == m]
            if not row.empty:
                matrix[i, j] = row["score"].values[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(dists, rotation=20, ha="right")
    ax.set_yticks(range(len(maps)))
    ax.set_yticklabels([short(m) for m in maps])
    ax.set_title(f"Throughput heatmap — range {kr:,}, read {int(ratio*100)}%, {threads}T")
    plt.colorbar(im, ax=ax, label=YLABEL)
    mv = matrix.max()
    for i in range(len(maps)):
        for j in range(len(dists)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if matrix[i,j] < mv*0.7 else "white")
    save_or_show(fig, save_dir, "heatmap", f"heatmap_range{kr}_r{ratio}_t{threads}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    save_dir = os.path.join(args.folder, "plots") if args.save else None

    print(f"Loading CSVs from {args.folder}...")
    df = load_folder(args.folder)

    threads   = sorted(df["threads"].unique())
    dists     = sorted(df["distribution"].unique())
    keyranges = sorted(df["keyrange"].unique())
    ratios    = sorted(df["readratio"].unique())
    t1        = threads[0]

    print(f"  Threads found:  {threads}")
    print(f"  Distributions:  {dists}")
    print(f"  Key ranges:     {keyranges}")
    print(f"  Read ratios:    {ratios}\n")

    print("Generating bar charts...")
    for dist in dists:
        for kr in keyranges:
            for ratio in ratios:
                plot_bar(df, t1, dist, kr, ratio, save_dir)

    print("Generating key range comparison...")
    for dist in dists:
        for ratio in ratios:
            plot_keyrange(df, t1, dist, ratio, save_dir)

    print("Generating distribution comparison...")
    for kr in keyranges:
        for ratio in ratios:
            plot_distribution(df, t1, kr, ratio, save_dir)

    print("Generating read/write ratio plots...")
    for dist in dists:
        for kr in keyranges:
            plot_readratio(df, t1, dist, kr, save_dir)

    print("Generating scalability plots...")
    for dist in dists:
        for kr in keyranges:
            for ratio in ratios:
                plot_scalability(df, dist, kr, ratio, save_dir)

    print("Generating heatmaps...")
    for kr in keyranges:
        for ratio in ratios:
            plot_heatmap(df, t1, kr, ratio, save_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()