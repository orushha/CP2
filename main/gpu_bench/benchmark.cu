// GPU Hash Table Benchmark — simple open-addressing implementation, no external deps
//
// Matches parameters of HashMapBenchmark.java exactly:
//   key ranges    : 1000, 1000000
//   read ratios   : 0.8, 0.5, 0.2
//   distributions : uniform, zipfian_0.5, zipfian_0.99
//   samples       : 20 per config (matching JMH 10 iterations × 2 forks)
//   pre-populate  : ~50% of key range, seed 42 (matching Java setup)
//
// Build:
//   nvcc -O2 -arch=sm_90  -o gpu_benchmark benchmark.cu   # H100
//   nvcc -O2 -arch=sm_80  -o gpu_benchmark benchmark.cu   # A100
//   nvcc -O2 -arch=sm_89  -o gpu_benchmark benchmark.cu   # RTX 6000 Ada / L40S
//
// Output CSV is format-compatible with JMH output (same column order).

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ── Hash table layout ────────────────────────────────────────────────────────
// Each slot holds (key, value) pair. EMPTY_KEY marks unused slots.
// Using int throughout; valid keys are [0, keyRange-1] so -1 is safe sentinel.
#define EMPTY_KEY (-1)

typedef struct { int key; int val; } KV;

// ── Device helpers ───────────────────────────────────────────────────────────

__device__ __forceinline__ uint32_t knuth_hash(int key, uint32_t cap_mask) {
    return ((uint32_t)key * 2654435761u) & cap_mask;
}

// Lock-free get: linear probing, no writes.
__device__ bool ht_get(const KV* __restrict__ table, uint32_t cap_mask,
                        int key, int* out) {
    uint32_t h = knuth_hash(key, cap_mask);
    uint32_t cap = cap_mask + 1;
    for (uint32_t i = 0; i < cap; i++) {
        uint32_t idx = (h + i) & cap_mask;
        // Plain load is fine for reads (GPU memory model: coherent after sync)
        int k = table[idx].key;
        if (k == EMPTY_KEY) return false;
        if (k == key) { *out = table[idx].val; return true; }
    }
    return false;
}

// CAS-based insert: atomically claim an empty slot.
__device__ void ht_put(KV* table, uint32_t cap_mask, int key, int val) {
    uint32_t h = knuth_hash(key, cap_mask);
    uint32_t cap = cap_mask + 1;
    for (uint32_t i = 0; i < cap; i++) {
        uint32_t idx = (h + i) & cap_mask;
        int old = atomicCAS(&table[idx].key, EMPTY_KEY, key);
        if (old == EMPTY_KEY || old == key) {
            atomicExch(&table[idx].val, val);
            return;
        }
    }
    // Table full — silently drop (won't happen with 2× capacity)
}

// ── Benchmark kernels ────────────────────────────────────────────────────────

// Each thread executes one operation: read (ops[tid]==0) or write (ops[tid]==1).
// 'sink' prevents dead-code elimination of reads; warp-level XOR reduction
// means only 1 atomic per 32 threads (not 1 per thread) so contention is minimal.
__global__ void bench_kernel(KV* __restrict__ table, uint32_t cap_mask,
                              const int* __restrict__ keys,
                              const int* __restrict__ ops,
                              int n,
                              int* __restrict__ sink) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int v = 0;
    if (tid < n) {
        int key = keys[tid];
        if (ops[tid] == 0)
            ht_get(table, cap_mask, key, &v);
        else
            ht_put(table, cap_mask, key, key);
    }
    // Warp-level XOR so only lane-0 of each warp touches global memory
    for (int off = 16; off > 0; off >>= 1)
        v ^= __shfl_down_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0)
        atomicXor(sink, v);
}

// Pre-populate kernel: insert keys[0..n-1] into table.
__global__ void prepop_kernel(KV* table, uint32_t cap_mask,
                               const int* __restrict__ keys, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    ht_put(table, cap_mask, keys[tid], keys[tid]);
}

// Reset kernel: write EMPTY_KEY to all slots.
__global__ void reset_kernel(KV* table, int cap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < cap) { table[tid].key = EMPTY_KEY; table[tid].val = 0; }
}

// ── CPU-side helpers ─────────────────────────────────────────────────────────

static uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// Simple LCG approximating Java's Random behaviour (good enough for distribution shape)
static uint64_t lcg_state;

static void lcg_seed(uint64_t seed) { lcg_state = seed; }

static double lcg_double() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((lcg_state >> 11) & 0x1FFFFFFFFFFFFFULL) / (double)(1ULL << 53);
}

static int lcg_int(int bound) {
    return (int)(lcg_double() * bound);
}

static void gen_uniform(int* out, int n, int key_range) {
    lcg_seed(42);
    for (int i = 0; i < n; i++) out[i] = lcg_int(key_range);
}

// Build Zipfian CDF for key_range ranks with given skew, then sample.
static void gen_zipfian(int* out, int n, int key_range, double skew) {
    double* cdf = (double*)malloc(key_range * sizeof(double));
    double sum = 0.0;
    for (int i = 1; i <= key_range; i++) sum += 1.0 / pow((double)i, skew);
    cdf[0] = (1.0 / pow(1.0, skew)) / sum;
    for (int i = 1; i < key_range; i++)
        cdf[i] = cdf[i-1] + (1.0 / pow((double)(i+1), skew)) / sum;

    lcg_seed(42);
    for (int i = 0; i < n; i++) {
        double u = lcg_double();
        // Binary search in CDF
        int lo = 0, hi = key_range - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cdf[mid] < u) lo = mid + 1; else hi = mid;
        }
        out[i] = lo;  // rank 0-based, matching Java's ZipfDistribution.sample()-1
    }
    free(cdf);
}

static void gen_prepop(int* out, int n, int key_range) {
    // Matches Java: new Random(42).nextInt(keyRange) for keyRange/2 inserts
    lcg_seed(42);
    for (int i = 0; i < n; i++) out[i] = lcg_int(key_range);
}

// ── Error checking ────────────────────────────────────────────────────────────
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ── Main ─────────────────────────────────────────────────────────────────────
int main() {
    const int   KEY_RANGES[]   = {1000, 1000000};
    const double READ_RATIOS[] = {0.8, 0.5, 0.2};
    const char* DISTS[]        = {"uniform", "zipfian_0.5", "zipfian_0.99"};
    const double SKEWS[]       = {0.0, 0.5, 0.99};  // 0.0 → uniform

    const int N_KEY_RANGES  = 2;
    const int N_READ_RATIOS = 3;
    const int N_DISTS       = 3;
    const int SAMPLE_SIZE   = 1000000;  // ops per kernel launch (= one JMH iteration)
    const int N_SAMPLES     = 20;       // launches per config (= JMH 10 iter × 2 forks)
    const int BLOCK_SIZE    = 256;

    // Print device info to stderr so stdout is clean CSV
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s  SM count: %d  Compute: %d.%d\n",
            prop.name, prop.multiProcessorCount,
            prop.major, prop.minor);
    fprintf(stderr, "SAMPLE_SIZE=%d  N_SAMPLES=%d  BLOCK_SIZE=%d\n",
            SAMPLE_SIZE, N_SAMPLES, BLOCK_SIZE);

    // CSV header — same columns as JMH output
    printf("\"Benchmark\",\"Mode\",\"Threads\",\"Samples\",\"Score\","
           "\"Score Error (99.9%%)\",\"Unit\","
           "\"Param: distribution\",\"Param: keyRange\","
           "\"Param: mapType\",\"Param: readRatio\"\n");

    // Allocate persistent GPU scratch for op results (Blackhole)
    int* d_sink;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(int)));

    // Allocate GPU ops array (same size for all configs)
    int* d_ops;
    CUDA_CHECK(cudaMalloc(&d_ops, SAMPLE_SIZE * sizeof(int)));

    // CPU op buffer
    int* cpu_ops = (int*)malloc(SAMPLE_SIZE * sizeof(int));

    for (int di = 0; di < N_DISTS; di++) {
        int* cpu_keys = (int*)malloc(SAMPLE_SIZE * sizeof(int));

        for (int ki = 0; ki < N_KEY_RANGES; ki++) {
            int key_range = KEY_RANGES[ki];

            // Generate key samples for this (dist, key_range) pair
            if (di == 0)
                gen_uniform(cpu_keys, SAMPLE_SIZE, key_range);
            else
                gen_zipfian(cpu_keys, SAMPLE_SIZE, key_range, SKEWS[di]);

            // Copy keys to GPU
            int* d_keys;
            CUDA_CHECK(cudaMalloc(&d_keys, SAMPLE_SIZE * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_keys, cpu_keys,
                                  SAMPLE_SIZE * sizeof(int), cudaMemcpyHostToDevice));

            // Pre-population keys: ~50% of key_range
            int prepop_n = key_range / 2;
            int* cpu_prepop = (int*)malloc(prepop_n * sizeof(int));
            gen_prepop(cpu_prepop, prepop_n, key_range);
            int* d_prepop;
            CUDA_CHECK(cudaMalloc(&d_prepop, prepop_n * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_prepop, cpu_prepop,
                                  prepop_n * sizeof(int), cudaMemcpyHostToDevice));

            // GPU hash table: capacity = next power of 2 ≥ 2 × key_range
            uint32_t cap = next_pow2((uint32_t)key_range * 2);
            uint32_t cap_mask = cap - 1;
            KV* d_table;
            CUDA_CHECK(cudaMalloc(&d_table, cap * sizeof(KV)));

            int prepop_blocks = (prepop_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int bench_blocks  = (SAMPLE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int reset_blocks  = (cap + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int ri = 0; ri < N_READ_RATIOS; ri++) {
                double read_ratio = READ_RATIOS[ri];

                // Build op type array: 0 = get, 1 = put/remove
                // Using a fresh seed per (di, ki, ri) to mirror JMH per-thread rng
                lcg_seed((uint64_t)(di * 100 + ki * 10 + ri + 7919));
                for (int i = 0; i < SAMPLE_SIZE; i++)
                    cpu_ops[i] = (lcg_double() < read_ratio) ? 0 : 1;
                CUDA_CHECK(cudaMemcpy(d_ops, cpu_ops,
                                      SAMPLE_SIZE * sizeof(int), cudaMemcpyHostToDevice));

                double scores[N_SAMPLES];

                for (int s = 0; s < N_SAMPLES; s++) {
                    // Reset table
                    reset_kernel<<<reset_blocks, BLOCK_SIZE>>>(d_table, (int)cap);

                    // Pre-populate (not timed)
                    prepop_kernel<<<prepop_blocks, BLOCK_SIZE>>>(
                        d_table, cap_mask, d_prepop, prepop_n);
                    CUDA_CHECK(cudaDeviceSynchronize());

                    // Timed benchmark
                    cudaEvent_t t0, t1;
                    CUDA_CHECK(cudaEventCreate(&t0));
                    CUDA_CHECK(cudaEventCreate(&t1));
                    CUDA_CHECK(cudaEventRecord(t0));
                    bench_kernel<<<bench_blocks, BLOCK_SIZE>>>(
                        d_table, cap_mask, d_keys, d_ops, SAMPLE_SIZE, d_sink);
                    CUDA_CHECK(cudaEventRecord(t1));
                    CUDA_CHECK(cudaEventSynchronize(t1));

                    float ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
                    scores[s] = (double)SAMPLE_SIZE / ((double)ms / 1000.0);

                    CUDA_CHECK(cudaEventDestroy(t0));
                    CUDA_CHECK(cudaEventDestroy(t1));
                }

                // Compute mean and 99.9% CI half-width (3.291σ / √n)
                double mean = 0.0;
                for (int s = 0; s < N_SAMPLES; s++) mean += scores[s];
                mean /= N_SAMPLES;
                double var = 0.0;
                for (int s = 0; s < N_SAMPLES; s++)
                    var += (scores[s] - mean) * (scores[s] - mean);
                double stdev = sqrt(var / (N_SAMPLES - 1));
                double err   = 3.291 * stdev / sqrt((double)N_SAMPLES);

                fprintf(stderr, "  [%s / kr=%d / rr=%.1f] mean=%.0f ops/s  err=%.0f\n",
                        DISTS[di], key_range, read_ratio, mean, err);

                // Output CSV row (Threads=1 represents 1 GPU device)
                printf("\"benchmarks.HashMapBenchmark.mixedReadWrite\","
                       "\"thrpt\",1,%d,%.6f,%.6f,\"ops/s\","
                       "%s,%d,GPUHashTable,%.1f\n",
                       N_SAMPLES, mean, err,
                       DISTS[di], key_range, read_ratio);
                fflush(stdout);
            }

            CUDA_CHECK(cudaFree(d_table));
            CUDA_CHECK(cudaFree(d_prepop));
            CUDA_CHECK(cudaFree(d_keys));
            free(cpu_prepop);
        }
        free(cpu_keys);
    }

    CUDA_CHECK(cudaFree(d_ops));
    CUDA_CHECK(cudaFree(d_sink));
    free(cpu_ops);
    return 0;
}
