#!/usr/bin/env bash
# Run JMH benchmarks for all thread counts appropriate to this machine.
# Saves each run to a timestamped CSV in app/results/.
# Usage:
#   ./run_experiements_offline.sh           # full benchmark
#   ./run_experiements_offline.sh --quick   # quick smoke test

set -e  # exit on any error

BENCHMARK="benchmarks.HashMapBenchmark"
RESULTS_DIR="app/results"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

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

mkdir -p "$RESULTS_DIR"

# Log system info
SYSINFO_FILE="$RESULTS_DIR/sysinfo-$TIMESTAMP.txt"
echo "Date: $TIMESTAMP" > "$SYSINFO_FILE"
echo "Hostname: $(hostname)" >> "$SYSINFO_FILE"
echo "Cores: $CORES" >> "$SYSINFO_FILE"
echo "OS: $(uname -a)" >> "$SYSINFO_FILE"
echo "Java: $(java -version 2>&1 | head -1)" >> "$SYSINFO_FILE"
echo "# System info saved to $SYSINFO_FILE"

# Smoke test first
echo ""
echo "# Running smoke test before full benchmark..."
./gradlew --offline jmh \
    -Pjmh.threads=1 \
    -Pjmh.include='benchmarks.QuickBenchmark' \
    -Pjmh.resultFormat='CSV' \
    -Pjmh.resultsFile="$RESULTS_DIR/quicktest-$TIMESTAMP.csv"

echo "# Smoke test passed! Starting full benchmark..."
echo ""

# Run benchmarks
for THREADS in "${THREAD_COUNTS[@]}"; do
    OUTPUT="$RESULTS_DIR/jmh-${TIMESTAMP}-threads${THREADS}.csv"
    echo ""
    echo "# [$THREADS threads] Starting... output -> $OUTPUT"

    ./gradlew --offline jmh \
        -Pjmh.threads="$THREADS" \
        -Pjmh.include="$BENCHMARK" \
        -Pjmh.resultFormat='CSV' \
        -Pjmh.resultsFile="$OUTPUT"

    echo "# [$THREADS threads] Done -> $OUTPUT"
done

echo ""
echo "# All runs complete. Results in $RESULTS_DIR:"
ls -lh "$RESULTS_DIR"/*.csv
