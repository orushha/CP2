#!/usr/bin/env bash
# Run JMH benchmarks for all thread counts appropriate to this machine.
# Saves each run to a timestamped CSV in app/results/.
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
RESULTS_DIR="app/results"
TIMESTAMP=$(date '+%d-%m-%Y_%H-%M-%S')

# Detect core count and pick thread counts accordingly
CORES=$(nproc)
echo "# Detected $CORES cores"

if [ "$1" == "--quick" ]; then
    BENCHMARK="benchmarks.QuickBenchmark"
    THREAD_COUNTS=(1 2 4)
    echo "# Mode: quick smoke test"
elif [ "$CORES" -le 4 ]; then
    # RPi5
    THREAD_COUNTS=(1 2 4 8)
    echo "# Mode: full benchmark (RPi5 profile)"
elif [ "$CORES" -le 48 ]; then
    # HPC node
    THREAD_COUNTS=(1 2 4 8 16 32 64)
    echo "# Mode: full benchmark (HPC profile)"
else
    # Fallback for unknown hardware
    THREAD_COUNTS=(1 2 4 8 16 32)
    echo "# Mode: full benchmark (generic profile, $CORES cores)"
fi

GRADLE="gradle --offline"
RUN_DIR="app/results/${HOSTNAME}-${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# Log system info (full runs only)
if [ "$1" != "--quick" ]; then
    SYSINFO_FILE="$RUN_DIR/sysinfo.txt"
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
$GRADLE jmh -Pquick -Pjmh.threads=1 \
    -Pjmh.resultFormat='CSV' \
    -Pjmh.resultsFile="$RESULTS_DIR/quicktest-$TIMESTAMP.csv"

echo "# Smoke test passed! Starting full benchmark..."
echo ""

# Run benchmarks
for THREADS in "${THREAD_COUNTS[@]}"; do
    OUTPUT="$RUN_DIR/threads${THREADS}.csv"
    echo ""
    echo "# [$THREADS threads] Starting... output -> $OUTPUT"

    $GRADLE jmh \
        -Pjmh.threads="$THREADS" \
        -Pjmh.resultsFile="$OUTPUT"

    echo "# [$THREADS threads] Done -> $OUTPUT"
done

echo ""
echo "# All runs complete. Results in $RESULTS_DIR:"
ls -lh "$RUN_DIR"/*.csv