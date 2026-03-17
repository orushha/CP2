package benchmarks;

import org.apache.commons.math3.distribution.ZipfDistribution;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Quick smoke test benchmark... uns in ~5 minutes.
 * Use this to verify everything works on a new machine before
 * committing to the full HashMapBenchmark run.
 */
@BenchmarkMode({Mode.Throughput, Mode.AverageTime})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(1)
@State(Scope.Benchmark)
public class QuickBenchmark {

    // Just the most representative implementations
    @Param({"SynchronizedMap", "StripedMap", "HashTrieMap", "WrapConcurrentHashMap"})
    public String mapType;

    // One representative workload
    @Param({"0.8", "0.2"})
    public double readRatio;

    // One key range
    @Param({"10000"})
    public int keyRange;

    // One distribution
    @Param({"uniform", "zipfian_0.99"})
    public String distribution;

    private static final int LOCK_COUNT = 32;
    private static final int SAMPLE_SIZE = 100_000; // smaller than full benchmark

    public OurMap<Integer, String> map;
    public int[] keys;

    @Setup(Level.Trial)
    public void setup() {
        switch (mapType) {
            case "SynchronizedMap":       map = new SynchronizedMap<>(); break;
            case "StripedMap":            map = new StripedMap<>(LOCK_COUNT); break;
            case "HashTrieMap":           map = new HashTrieMap<>(); break;
            case "WrapConcurrentHashMap": map = new WrapConcurrentHashMap<>(); break;
            default: throw new IllegalArgumentException("Unknown map: " + mapType);
        }

        keys = generateKeys(distribution, keyRange, SAMPLE_SIZE);

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
                samples[i] = zipf.sample() - 1;
        }
        return samples;
    }

    @State(Scope.Thread)
    public static class ThreadState {
        public Random random;
        public int index;

        @Setup(Level.Trial)
        public void setup() {
            random = new Random(Thread.currentThread().getId());
            index = (int)(Thread.currentThread().getId() % 100_000);
        }
    }

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