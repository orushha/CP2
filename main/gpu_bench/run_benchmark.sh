#!/bin/bash
#SBATCH --job-name=gpu_hashmap
#SBATCH --output=gpu_benchmark_%j.out
#SBATCH --error=gpu_benchmark_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=dgx1

# load CUDA
module load CUDA/12.1.1

# build
mkdir -p build
cd build
cmake ..
make

# run benchmark
./gpu_benchmark

# copy results to results/gpu/
cp gpu_results.csv ../../../../results/gpu/gpu_results.csv

echo "done!"