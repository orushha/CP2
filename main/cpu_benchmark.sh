#!/usr/bin/env bash
# Run JMH benchmarks for all thread counts appropriate to this machine.
# All runs share one timestamped results directory; filenames encode thread count.
#
# Usage:
#   ./run_benchmarks.sh            # full benchmark
#   ./run_benchmarks.sh --quick    # quick smoke test
#
# First-time setup (needs internet, run from a login node):
#   ./setup_deps.sh
# All subsequent runs are fully offline.
set -euo pipefail

# ── Locate Gradle ─────────────────────────────────────────────────────────────
# Prefer the Gradle wrapper committed to the repo (most portable).
# Fall back to: system gradle → common HPC module locations.
find_gradle() {
    if [ -x "./gradlew" ]; then
        echo "./gradlew"
        return
    fi
    if command -v gradle &>/dev/null; then
        echo "gradle"
        return
    fi
    # HPC: try loading a Java module then check again
    if command -v module &>/dev/null; then
        # Adjust the module name to whatever your cluster provides
        module load java/21 2>/dev/null || true
        if command -v gradle &>/dev/null; then
            echo "gradle"
            return
        fi
    fi
    # DGX / bare-metal fallback: common manual install paths
    for candidate in \
        /opt/gradle/bin/gradle \
        "$HOME/.sdkman/candidates/gradle/current/bin/gradle"
    do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done
    echo "ERROR: gradle not found. Run setup_deps.sh or install Gradle." >&2
    exit 1
}

GRADLE="$(find_gradle) --offline"
echo "# Using Gradle: $GRADLE"

# ── Parse arguments ───────────────────────────────────────────────────────────
QUICK_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK_FLAG="-Pquick" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ── Detect core count and pick thread counts ──────────────────────────────────
CORES=$(nproc)
echo "# Detected $CORES cores"

if [ "$CORES" -le 4 ]; then
    THREAD_COUNTS=(1 2 4 8)
    echo "# Mode: RPi5 profile"
elif [ "$CORES" -le 48 ]; then
    THREAD_COUNTS=(1 2 4 8 16 32 64)
    echo "# Mode: HPC profile"
else
    # DGX A100/H100 has 128 cores; include higher thread counts
    # BUG FIX: original script had these two profiles swapped
    THREAD_COUNTS=(1 2 4 8 16 32 64 128)
    echo "# Mode: DGX profile ($CORES cores)"
fi

# Quick mode overrides to a single thread count for a fast smoke test
if [ -n "$QUICK_FLAG" ]; then
    THREAD_COUNTS=(1)
    echo "# Quick mode: single run, t=1"
fi

# ── Shared run directory ──────────────────────────────────────────────────────
# BUG FIX: RUN_DIR was never assigned before; every call passed -Prun.dir=""
HOSTNAME_CLEAN=$(hostname -s)                           # short hostname, no domain
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
RUN_DIR="${HOSTNAME_CLEAN}-${TIMESTAMP}"
RESULTS_BASE="../results/${RUN_DIR}"

mkdir -p "$RESULTS_BASE"
echo "# Results directory: $RESULTS_BASE"
echo ""

# ── Run benchmarks ─────────────────────────────────────────────────────────────
for THREADS in "${THREAD_COUNTS[@]}"; do
    echo "# [$THREADS threads] Starting..."
    $GRADLE jmh \
        -Pjmh.threads="$THREADS" \
        -Prun.dir="$RUN_DIR" \
        ${QUICK_FLAG}
    echo "# [$THREADS threads] Done"
    echo ""
done

echo "# All runs complete. Results in ${RESULTS_BASE}:"
ls -lh "$RESULTS_BASE"