package benchmarks;

import org.apache.commons.math3.distribution.ZipfDistribution;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode({Mode.Throughput, Mode.AverageTime})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 10, time = 2)
@Fork(2)
@State(Scope.Benchmark)  // ADDED — required because @Param fields live here
public class HashMapBenchmark {

    // --- Parameters ---

    @Param({"SynchronizedMap", "StripedMap", "StripedMapPadded",
            "StripedWriteMap", "StripedWriteMapPadded",
            "StripedLevelWriteMap", "HashTrieMap", "WrapConcurrentHashMap"})
    public String mapType;

    @Param({"0.8", "0.5", "0.2"})
    public double readRatio;

    @Param({"1000", "1000000"})
    public int keyRange;

    @Param({"uniform", "zipfian_0.5", "zipfian_0.99"})
    public String distribution;

    private static final int LOCK_COUNT = 32;
    private static final int SAMPLE_SIZE = 1_000_000;

    // --- Shared map + precomputed keys ---

    public OurMap<Integer, String> map;
    public int[] keys;

    @Setup(Level.Trial)
    public void setup() {
        switch (mapType) {
            case "SynchronizedMap":       map = new SynchronizedMap<>(); break;
            case "StripedMap":            map = new StripedMap<>(LOCK_COUNT); break;
            case "StripedMapPadded":      map = new StripedMapPadded<>(LOCK_COUNT); break;
            case "StripedWriteMap":       map = new StripedWriteMap<>(LOCK_COUNT); break;
            case "StripedWriteMapPadded": map = new StripedWriteMapPadded<>(LOCK_COUNT); break;
            case "StripedLevelWriteMap":  map = new StripedLevelWriteMap<>(LOCK_COUNT); break;
            case "HashTrieMap":           map = new HashTrieMap<>(); break;
            case "WrapConcurrentHashMap": map = new WrapConcurrentHashMap<>(); break;
            default: throw new IllegalArgumentException("Unknown map: " + mapType);
        }

        // Pre-compute key samples to keep hot path clean
        keys = generateKeys(distribution, keyRange, SAMPLE_SIZE);

        // Pre-populate ~50% of key range so reads have something to find
        Random rng = new Random(42);
        for (int i = 0; i < keyRange / 2; i++) {
            int k = rng.nextInt(keyRange);
            map.put(k, Integer.toString(k));
        }
    }

    private static int[] generateKeys(String dist, int range, int count) {
        int[] samples = new int[count];
        if (dist.equals("uniform")) {
            Random rng = new Random(42);
            for (int i = 0; i < count; i++)
                samples[i] = rng.nextInt(range);
        } else {
            double skew = dist.equals("zipfian_0.5") ? 0.5 : 0.99;
            ZipfDistribution zipf = new ZipfDistribution(range, skew);
            zipf.reseedRandomGenerator(42);
            for (int i = 0; i < count; i++)
                samples[i] = zipf.sample() - 1; // ZipfDistribution is 1-indexed
        }
        return samples;
    }

    // --- Thread-local state ---

    @State(Scope.Thread)
    public static class ThreadState {
        public Random random;
        public int index;

        @Setup(Level.Trial)
        public void setup() {
            random = new Random(Thread.currentThread().getId());
            index = (int)(Thread.currentThread().getId() % SAMPLE_SIZE);
        }
    }

    // --- Benchmark ---

    @Benchmark
    public void mixedReadWrite(ThreadState ts, Blackhole bh) {
        int key = keys[ts.index % SAMPLE_SIZE];
        ts.index++;

        if (ts.random.nextDouble() < readRatio) {
            bh.consume(map.get(key));
        } else {
            if (ts.random.nextBoolean()) {
                map.put(key, Integer.toString(key));
            } else {
                map.remove(key);
            }
        }
    }
}