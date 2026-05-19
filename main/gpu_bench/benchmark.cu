#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cmath>
#include <numeric>

// same params as HashMapBenchmark.java
const int SAMPLE_SIZE  = 5000000; // 5M ops: ~1.7ms/sample at GPU speeds, reduces timing noise
const int NUM_WARMUP   = 5;       // discarded — lets CUDA JIT and caches warm up
const int NUM_SAMPLES  = 20;

const std::vector<int>         KEY_RANGES    = {1000, 1000000};
const std::vector<double>      READ_RATIOS   = {0.8, 0.5, 0.2};
const std::vector<std::string> DISTRIBUTIONS = {
    "uniform", "zipfian_0.5", "zipfian_0.99"
};

// uniform random keys, seed 42 for reproducibility
std::vector<int> generate_uniform(int count, int key_range, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, key_range - 1);
    std::vector<int> keys(count);
    for (auto& k : keys) k = dist(rng);
    return keys;
}

// zipfian keys - higher skew = more skewed access pattern
std::vector<int> generate_zipfian(int count, int key_range,
                                   double skew, int seed = 42) {
    std::mt19937 rng(seed);

    std::vector<double> probs(key_range);
    double sum = 0.0;
    for (int i = 1; i <= key_range; i++) {
        probs[i-1] = 1.0 / std::pow((double)i, skew);
        sum += probs[i-1];
    }
    for (auto& p : probs) p /= sum;

    // build CDF for inverse sampling
    std::vector<double> cdf(key_range);
    cdf[0] = probs[0];
    for (int i = 1; i < key_range; i++)
        cdf[i] = cdf[i-1] + probs[i];

    std::uniform_real_distribution<double> udist(0.0, 1.0);
    std::vector<int> keys(count);
    for (auto& k : keys) {
        double u = udist(rng);
        int idx = std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
        k = std::min(idx, key_range - 1);
    }
    return keys;
}

std::vector<int> generate_keys(const std::string& dist,
                                int key_range, int count) {
    if (dist == "uniform")
        return generate_uniform(count, key_range);
    else if (dist == "zipfian_0.5")
        return generate_zipfian(count, key_range, 0.5);
    else
        return generate_zipfian(count, key_range, 0.99);
}

// runs one benchmark configuration and returns ops/sec
double run_single(const std::vector<int>& keys, int key_range,
                  double read_ratio, int seed) {

    const int EMPTY_KEY   = -1;
    const int ERASED_KEY  = -2;
    const int EMPTY_VAL   = -1;
    const std::size_t capacity = static_cast<std::size_t>(key_range) * 2;

    // current cuco API uses cuco::extent for the capacity argument;
    // erased_key must differ from empty_key when erase() is used
    cuco::static_map<int, int> map{
        cuco::extent<std::size_t>{capacity},
        cuco::empty_key{EMPTY_KEY},
        cuco::empty_value{EMPTY_VAL},
        cuco::erased_key{ERASED_KEY}
    };

    // pre-populate ~50% of key range so reads have something to find
    std::mt19937 pre_rng(42);
    std::uniform_int_distribution<int> pre_dist(0, key_range - 1);
    std::vector<cuco::pair<int,int>> init_pairs;
    init_pairs.reserve(key_range / 2);
    for (int i = 0; i < key_range / 2; i++) {
        int k = pre_dist(pre_rng);
        init_pairs.push_back({k, k});
    }
    thrust::device_vector<cuco::pair<int,int>> d_init(init_pairs);
    map.insert(d_init.begin(), d_init.end());

    // split ops: read_ratio -> get, rest split between put and remove
    std::mt19937 op_rng(seed);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);

    std::vector<int> get_keys, put_keys, remove_keys;
    get_keys.reserve(SAMPLE_SIZE);
    put_keys.reserve(SAMPLE_SIZE / 4);
    remove_keys.reserve(SAMPLE_SIZE / 4);

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        int key = keys[i % keys.size()];
        double op = op_dist(op_rng);
        if (op < read_ratio)
            get_keys.push_back(key);
        else if (op_rng() % 2 == 0)
            put_keys.push_back(key);
        else
            remove_keys.push_back(key);
    }

    // move to GPU
    thrust::device_vector<int> d_get_keys(get_keys);
    thrust::device_vector<int> d_get_results(get_keys.size());
    thrust::device_vector<int> d_remove_keys(remove_keys);

    thrust::device_vector<int> d_put_keys(put_keys);
    thrust::device_vector<cuco::pair<int,int>> d_put_pairs(put_keys.size());
    thrust::transform(
        d_put_keys.begin(), d_put_keys.end(),
        d_put_pairs.begin(),
        [] __device__ (int k) {
            return cuco::pair<int,int>{k, k};
        }
    );

    // sync before timing to make sure data transfer is done
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    map.find(d_get_keys.begin(), d_get_keys.end(),
             d_get_results.begin());

    map.insert(d_put_pairs.begin(), d_put_pairs.end());

    map.erase(d_remove_keys.begin(), d_remove_keys.end());

    // sync after so we capture actual GPU completion time
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(end - start).count();
    return SAMPLE_SIZE / elapsed_s;
}

int main() {

  // print GPU name so we know which device ran the benchmark
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "# GPU: " << prop.name << std::endl;

    // output CSV in same format as JMH so compare_hardware.py works
    std::ofstream csv("gpu_results.csv");
    csv << "\"Benchmark\",\"Mode\",\"Threads\",\"Samples\","
        << "\"Score\",\"Score Error (99.9%)\",\"Unit\","
        << "\"Param: distribution\",\"Param: keyRange\","
        << "\"Param: mapType\",\"Param: readRatio\"\n";

    for (int key_range : KEY_RANGES) {
        for (double read_ratio : READ_RATIOS) {
            for (const auto& dist : DISTRIBUTIONS) {

                std::cout << "keyRange=" << key_range
                          << " readRatio=" << read_ratio
                          << " dist=" << dist << " ";

                // generate keys once per config, same as @Setup(Level.Trial)
                std::vector<int> keys = generate_keys(
                    dist, key_range, SAMPLE_SIZE
                );

                for (int w = 0; w < NUM_WARMUP; w++)
                    run_single(keys, key_range, read_ratio, -(w + 1));

                std::vector<double> scores;
                scores.reserve(NUM_SAMPLES);
                for (int s = 0; s < NUM_SAMPLES; s++) {
                    scores.push_back(
                        run_single(keys, key_range, read_ratio, s)
                    );
                    std::cout << "." << std::flush;
                }
                std::cout << " done" << std::endl;

                // mean and 99.9% confidence interval
                double mean = 0.0;
                for (double s : scores) mean += s;
                mean /= NUM_SAMPLES;

                double variance = 0.0;
                for (double s : scores)
                    variance += (s - mean) * (s - mean);
                variance /= (NUM_SAMPLES - 1);
                double error = 3.291 * std::sqrt(variance / NUM_SAMPLES);

                csv << "\"gpu.HashMapBenchmark.mixedReadWrite\","
                    << "\"thrpt\","
                    << "1,"
                    << NUM_SAMPLES << ","
                    << std::fixed << std::setprecision(6)
                    << mean << ","
                    << error << ","
                    << "\"ops/s\","
                    << dist << ","
                    << key_range << ","
                    << "cuco_static_map" << ","
                    << read_ratio << "\n";
            }
        }
    }

    csv.close();
    std::cout << "# done, results in gpu_results.csv" << std::endl;
    return 0;
}
