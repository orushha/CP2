#!/usr/bin/env bash
# Run JMH benchmarks for all thread counts appropriate to this machine.
# Saves each run to a timestamped CSV in app/results/<hostname>/.
# Usage:
#   ./run_benchmarks.sh                 # full benchmark
#   ./run_benchmarks.sh --quick         # quick smoke test
#
#   TO RUN OFFLINE:
#   gradle jmh -Pquick -Pjmh.threads=1  # first time, needs internet
#   ./run_benchmarks.sh                 # all subsequent runs, fully offline
set -e  # exit on any error

# If gradle isn't on PATH, re-launch inside nix dev shell automatically
if ! command -v gradle &> /dev/null; then
    echo "# gradle not found, entering nix dev shell..."
    exec nix develop --command bash "$0" "$@"
fi

BENCHMARK="benchmarks.HashMapBenchmark"
TIMESTAMP=$(date '+%d-%m-%Y_%H-%M-%S')

# Detect core count and pick thread counts accordingly
CORES=$(nproc)
echo "# Detected $CORES cores"

if [ "$1" == "--quick" ]; then
    BENCHMARK="benchmarks.QuickBenchmark"
    THREAD_COUNTS=(1 2 4)
    echo "# Mode: quick smoke test"
elif [ "$CORES" -le 4 ]; then
    THREAD_COUNTS=(1 2 4 8)
    echo "# Mode: full benchmark (RPi5 profile)"
elif [ "$CORES" -le 48 ]; then
    THREAD_COUNTS=(1 2 4 8 16 32 64)
    echo "# Mode: full benchmark (HPC profile)"
else
    THREAD_COUNTS=(1 2 4 8 16 32)
    echo "# Mode: full benchmark (generic profile, $CORES cores)"
fi

GRADLE="gradle --offline"

# Session folder — passed into Gradle as run.dir; Gradle owns the full output path
RUN_DIR="$(hostname)-${TIMESTAMP}"
mkdir -p "app/results/$RUN_DIR"

# Log system info (full runs only)
if [ "$1" != "--quick" ]; then
    SYSINFO_FILE="app/results/$RUN_DIR/sysinfo.txt"
    echo "Date: $TIMESTAMP" > "$SYSINFO_FILE"
    echo "Hostname: $(hostname)" >> "$SYSINFO_FILE"
    echo "Cores: $CORES" >> "$SYSINFO_FILE"
    echo "OS: $(uname -a)" >> "$SYSINFO_FILE"
    echo "Java: $(java -version 2>&1 | head -1)" >> "$SYSINFO_FILE"
    echo "# System info saved to $SYSINFO_FILE"
fi

# Smoke test first
echo ""
echo "# Running smoke test before full benchmark..."
$GRADLE jmh -Pquick -Pjmh.threads=1
echo "# Smoke test passed! Starting full benchmark..."
echo ""

# Run benchmarks
for THREADS in "${THREAD_COUNTS[@]}"; do
    echo "# [$THREADS threads] Starting..."
    $GRADLE jmh \
        -Pjmh.threads="$THREADS" \
        -Prun.dir="$RUN_DIR"
    echo "# [$THREADS threads] Done"
done

echo ""
echo "# All runs complete. Results in app/results/$RUN_DIR:"
ls -lh "app/results/$RUN_DIR"