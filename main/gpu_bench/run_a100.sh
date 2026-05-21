#!/bin/bash
#SBATCH --job-name=gpu_hashtable
#SBATCH --output=gpu_a100_%j.out
#SBATCH --error=gpu_a100_%j.err
#SBATCH --nodelist=cn13
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=scavenge
# A100 40 GiB (SM_80).

set -euo pipefail

echo "=== GPU Hash Table Benchmark (A100) ==="
echo "Host  : $(hostname)"
echo "Date  : $(date)"
echo "Job   : $SLURM_JOB_ID"

module load CUDA/12.1.1

echo "CUDA  : $(nvcc --version | grep release)"
nvidia-smi --query-gpu=name,memory.total,compute_cap \
           --format=csv,noheader

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/main/gpu_bench"
RESULTS_DIR="$REPO_ROOT/results/gpu"
mkdir -p "$RESULTS_DIR"

cd "$SCRIPT_DIR"
make SM=80

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
HOST_SHORT=$(hostname -s)
OUTFILE="$RESULTS_DIR/${HOST_SHORT}-${TIMESTAMP}.csv"

echo "=== Running benchmark, writing CSV to $OUTFILE ==="
./gpu_benchmark > "$OUTFILE"

echo "=== Done ==="
echo "Result file: $OUTFILE"
wc -l "$OUTFILE"
