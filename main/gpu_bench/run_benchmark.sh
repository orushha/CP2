#!/bin/bash
#SBATCH --job-name=gpu_hashmap
#SBATCH --output=gpu_benchmark_%j.out
#SBATCH --error=gpu_benchmark_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=scavenge
#SBATCH --constraint="gpu_v100|gpu_a30|gpu_a100_40gb|gpu_a100_80gb|gpu_h100"

# load modules
module load CUDA/12.1.1
module load CMake/3.26.3-GCCcore-12.3.0

# build
mkdir -p build
cd build
cmake ..
make -j4

# run benchmark
./gpu_benchmark

# To copy results to results/gpu/
cp gpu_results.csv ../../../../results/gpu/gpu_results.csv

echo "done!"
