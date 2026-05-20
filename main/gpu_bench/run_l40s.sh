#!/bin/bash
#SBATCH --job-name=gpu_hashmap
#SBATCH --output=gpu_benchmark_%j.out
#SBATCH --error=gpu_benchmark_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn19
#SBATCH --partition=scavenge
#SBATCH --time=02:00:00
# cn19: 4x NVIDIA L40S 48GiB (SM_89, Ada Lovelace)

set -euo pipefail

echo "=== GPU Hashmap Benchmark ==="
echo "Running on: $(hostname)"
echo "Date: $(date)"

module load CUDA/12.1.1
echo "nvcc: $(nvcc --version | head -1)"

export PATH="$HOME/.local/bin:$PATH"
echo "cmake: $(cmake --version | head -1)"

export CUCO_SOURCE_DIR="$HOME/cpm_cache/cuCollections-v0.0.1"

REPO_ROOT="$HOME/CP2"
SCRIPT_DIR="$REPO_ROOT/main/gpu_bench"
BUILD_DIR="$SCRIPT_DIR/build_${SLURM_JOB_ID}"
RESULTS_DIR="$REPO_ROOT/results/gpu"

mkdir -p "$BUILD_DIR" "$RESULTS_DIR"
cd "$BUILD_DIR"

echo "=== Configuring ==="
cmake "$SCRIPT_DIR" -DCMAKE_CUDA_ARCHITECTURES=89

echo "=== Building ==="
make -j"$(nproc)"

echo "=== Running benchmark ==="
./gpu_benchmark

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTFILE="$RESULTS_DIR/$(hostname)-${TIMESTAMP}.csv"
cp gpu_results.csv "$OUTFILE"
echo "=== Results saved to $OUTFILE ==="

cd "$REPO_ROOT" && rm -rf "$BUILD_DIR"
