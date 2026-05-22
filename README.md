# CP2: CPU Concurrent Hash Tables vs. GPU-Accelerated Hash Table

**Authors:** Orusha Thapa Magar (orth@itu.dk), Vaiva Staugaityte (vais@itu.dk)
**Course:** Computer Systems Performance, Spring 2026

## Structure

```
results/cpu/       JMH benchmark results from DGX Spark CPU (7 thread counts)
results/gpu/       GPU benchmark results from ITU HPC cn13 (NVIDIA A100)
results/figures/   Generated figures (run compare_hardware.py to regenerate)
main/app/          Java concurrent hash table implementations and JMH benchmark
main/gpu_bench/    CUDA GPU hash table benchmark (benchmark.cu)
main/compare_hardware.py   Analysis and figure generation script
```

## Dependencies

- Java 21, Gradle (CPU benchmarks, fully offline via `main/local-plugin-repo/`)
- CUDA 12.1.1, nvcc (GPU benchmark)
- Python 3 with pandas, matplotlib, numpy (figure generation)
