// Measuring concurrent performance of various threadsafe hash map
// implementations. All now perform bucket array reallocation internally. 

// sestoft@itu.dk * Based on TestStripedMapSolution.java 2014-10-07, 2025-06-26

import java.util.Locale;
import java.util.Random;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;
import java.util.concurrent.atomic.AtomicInteger;

public class HashMapPerformance {
  public static void main(String[] args) {
    SystemInfo();
    timeAllMaps(16);
    timeAllMapsRange();
  }

  private static void timeAllMapsRange() {
    for (int threadCount=1; threadCount<=32; threadCount++) 
      timeAllMaps(threadCount);
  }

  private static void timeAllMaps(int threadCount) {
    final int lockCount = 32, iterations = 5_000_000, range = 50_000,
      perThread = iterations / threadCount;
    final String format = "%-21s %3d %8d %5.2f %5.2f",
      header = String.format("%-21s %3s %8s %5s %5s", "implementation", "thr", "range", "add", "rem");
    if (threadCount == 1)
      System.out.printf("%-45s %15s us %10s %10s%n", header, "mean", "sdev", "repeats");
    for (double a : new double[] { 0.30, 0.60, 0.90 }) { // Probability of adding missing key
      for (double r : new double[] { 0.02, 0.10, 0.30 }) { // Probability of removing present key
	final double addProb = a, removeProb = r;
	Function<String, String> info =
	  s -> String.format(Locale.US, format, s, threadCount, range, addProb, removeProb);
	Mark7(info.apply("SynchronizedMap"),
	      i -> exerciseMap(threadCount, perThread, range, addProb, removeProb,
			       new SynchronizedMap<Integer,String>()));
	Mark7(info.apply("StripedMap"),
	      i -> exerciseMap(threadCount, perThread, range, addProb, removeProb,
			       new StripedMap<Integer,String>(lockCount)));
	Mark7(info.apply("StripedMapPadded"),
	      i -> exerciseMap(threadCount, perThread, range, addProb, removeProb,
			       new StripedMapPadded<Integer,String>(lockCount)));
	Mark7(info.apply("StripedWriteMap"), 
	      i -> exerciseMap(threadCount, perThread, range, addProb, removeProb,
			       new StripedWriteMap<Integer,String>(lockCount)));
	Mark7(info.apply("StripedWriteMapPadded"), 
	      i -> exerciseMap(threadCount, perThread, range, addProb, removeProb,
			       new StripedWriteMapPadded<Integer,String>(lockCount)));
	Mark7(info.apply("StripedLevelWriteMap"), 
	      i -> exerciseMap(threadCount, perThread, range, addProb, removeProb,
			       new StripedLevelWriteMap<Integer,String>(lockCount)));
	Mark7(info.apply("HashTrieMap"),
	      i -> exerciseMap(threadCount, perThread, range, addProb, removeProb,
			       new HashTrieMap<Integer,String>()));
	Mark7(info.apply("WrapConcHashMap"),
	      i -> exerciseMap(threadCount, perThread, range,addProb, removeProb,
			       new WrapConcurrentHashMap<Integer,String>()));
      }
    }
  }

  private static double exerciseMap(int threadCount, int perThread, int range,
				    double addProb, double removeProb,
                                    final OurMap<Integer, String> map) {
    Thread[] threads = new Thread[threadCount];
    for (int t=0; t<threadCount; t++) {
      final int myThread = t;
      threads[t] = new Thread(() -> {
        Random random = new Random(37 * myThread + 78);
        for (int i=0; i<perThread; i++) {
          Integer key = random.nextInt(range);
          if (!map.containsKey(key)) {
            // Add key with probability addProb
            if (random.nextDouble() < addProb) 
              map.put(key, Integer.toString(key));
          } 
          else // Remove key with probability removeProb
            if (random.nextDouble() < removeProb) 
              map.remove(key);
        }
        final AtomicInteger ai = new AtomicInteger();
        map.forEach((Integer k, String v) -> { ai.getAndIncrement(); });
        // System.out.println(ai.intValue() + " " + map.size());
      });
    }
    for (int t=0; t<threadCount; t++) 
      threads[t].start();
    try {
      for (int t=0; t<threadCount; t++) 
        threads[t].join();
    } catch (InterruptedException exn) { }
    return map.size();
  }

  // --- Benchmarking infrastructure ---

  // Modified to show microseconds instead of nanoseconds

  public static double Mark7(String msg, IntToDoubleFunction f) {
    int n = 10, count = 1, totalCount = 0;
    double dummy = 0.0, runningTime = 0.0, st = 0.0, sst = 0.0;
    do { 
      count *= 2;
      st = sst = 0.0;
      for (int j=0; j<n; j++) {
        Timer t = new Timer();
        for (int i=0; i<count; i++) 
          dummy += f.applyAsDouble(i);
        runningTime = t.check();
        double time = runningTime * 1e6 / count; // microseconds
        st += time; 
        sst += time * time;
        totalCount += count;
      }
    } while (runningTime < 0.25 && count < Integer.MAX_VALUE/2);
    double mean = st/n, sdev = Math.sqrt((sst - mean*mean*n)/(n-1));
    System.out.printf(Locale.US, "%-45s %15.1f us %10.2f %10d%n", msg, mean, sdev, count);
    return dummy / totalCount;
  }

  public static void SystemInfo() {
    System.out.printf("# OS:   %s; %s; %s%n", 
                      System.getProperty("os.name"), 
                      System.getProperty("os.version"), 
                      System.getProperty("os.arch"));
    System.out.printf("# JVM:  %s; %s%n", 
                      System.getProperty("java.vendor"), 
                      System.getProperty("java.version"));
    // The processor identifier works only on MS Windows:
    System.out.printf("# CPU:  %s; %d \"cores\"%n", 
		      System.getenv("PROCESSOR_IDENTIFIER"),
		      Runtime.getRuntime().availableProcessors());
    java.util.Date now = new java.util.Date();
    System.out.printf("# Date: %s%n", 
      new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ").format(now));
  }
}
