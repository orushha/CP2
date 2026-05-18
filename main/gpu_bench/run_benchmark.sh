#!/bin/bash
#SBATCH --job-name=gpu_hashmap
#SBATCH --output=gpu_benchmark_%j.out
#SBATCH --error=gpu_benchmark_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --time=02:00:00
#SBATCH --partition=scavenge
# For longer dedicated runs (3 days), change partition to: acltr
# For H100 instead of A100, change gres to: gpu:h100:1

set -euo pipefail

echo "=== GPU Hashmap Benchmark ==="
echo "Running on: $(hostname)"
echo "Date: $(date)"

# CUDA from the module system (no cmake here, so no toolchain conflict)
module load CUDA/12.1.1
echo "nvcc: $(nvcc --version | head -1)"

# cmake from pip install --user (run once on login node: pip install cmake --user)
export PATH="$HOME/.local/bin:$PATH"
echo "cmake: $(cmake --version | head -1)"

# point to pre-cloned cuCollections (cloned on login node, avoids internet on compute)
# if not pre-cloned, cmake falls back to downloading via CPM
export CUCO_SOURCE_DIR="$HOME/cpm_cache/cuCollections-v0.0.1"

REPO_ROOT="$HOME/CP2"
SCRIPT_DIR="$REPO_ROOT/main/gpu_bench"
BUILD_DIR="$SCRIPT_DIR/build"
RESULTS_DIR="$REPO_ROOT/results/gpu"

mkdir -p "$BUILD_DIR" "$RESULTS_DIR"
cd "$BUILD_DIR"

echo "=== Configuring ==="
# SM_80 = A100. Pass -DCMAKE_CUDA_ARCHITECTURES=90 for H100.
cmake "$SCRIPT_DIR" -DCMAKE_CUDA_ARCHITECTURES=80

echo "=== Building ==="
make -j"$(nproc)"

echo "=== Running benchmark ==="
./gpu_benchmark

# name the output file like the CPU results: hostname-timestamp.csv
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTFILE="$RESULTS_DIR/$(hostname)-${TIMESTAMP}.csv"
cp gpu_results.csv "$OUTFILE"
echo "=== Results saved to $OUTFILE ==="
