// Sequential and concurrent test of hash map implementations
// sestoft@itu.dk * 2014-10-21, 2025-05-27

// Based on 2014 TestStripedMapTestSolution.java

// Run with assertions enabled for sequential functional test of the maps:
//   java -ea HashMapTest

import java.util.Random;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;

public class HashMapTest {
  public static void main(String[] args) throws Exception {
    SystemInfo();
    testAllMapsSequential();
    testAllMapsConcurrent();
  }

  private static void testAllMapsSequential() {
    testMapSequential(new SynchronizedMap<Integer,String>());
    testMapSequential(new StripedMap<Integer,String>(5));
    testMapSequential(new StripedWriteMap<Integer,String>(5));
    testMapSequential(new StripedWriteMapPadded<Integer,String>(5));
    testMapSequential(new StripedLevelWriteMap<Integer,String>(5));
    testMapSequential(new HashTrieMap<Integer,String>());
    testMapSequential(new WrapConcurrentHashMap<Integer,String>());
  }

  private static void testAllMapsConcurrent() throws Exception {
    final int lockCount = 7, threadCount = 16;
    final int perThread = 1_000_000;
    final int range = 100;
    testMapConcurrent(threadCount, perThread, range,
     		      new SynchronizedMap<Integer,String>());
    testMapConcurrent(threadCount, perThread, range,
     		      new StripedMap<Integer,String>(lockCount));
    testMapConcurrent(threadCount, perThread, range,
     		      new StripedMapPadded<Integer,String>(lockCount));
    testMapConcurrent(threadCount, perThread, range,
		      new StripedWriteMap<Integer,String>(lockCount));
    testMapConcurrent(threadCount, perThread, range,
		      new StripedWriteMapPadded<Integer,String>(lockCount));
    testMapConcurrent(threadCount, perThread, range,
		      new StripedLevelWriteMap<Integer,String>(lockCount));
    testMapConcurrent(threadCount, perThread, range,
     		      new HashTrieMap<Integer,String>());
    testMapConcurrent(threadCount, perThread, range,
		      new WrapConcurrentHashMap<Integer,String>());
  }

  private static void testMapSequential(final OurMap<Integer, String> map) {
    System.out.printf("%nSequential test: %s%n", map.getClass());
    assert map.size() == 0;
    assert !map.containsKey(117);
    assert !map.containsKey(-2);
    assert map.get(117) == null;
    assert map.put(117, "A") == null;
    assert map.containsKey(117);
    assert map.get(117).equals("A");
    assert map.put(17, "B") == null;
    assert map.size() == 2;
    assert map.containsKey(17);
    assert map.get(117).equals("A");
    assert map.get(17).equals("B");
    assert map.put(117, "C").equals("A");
    assert map.containsKey(117);
    assert map.get(117).equals("C");
    assert map.size() == 2;
    map.forEach((k, v) -> System.out.printf("%10d maps to %s%n", k, v));
    assert map.remove(117).equals("C");
    assert !map.containsKey(117);
    assert map.get(117) == null;
    assert map.size() == 1;
    // assert map.putIfAbsent(17, "D").equals("B");
    assert map.get(17).equals("B");
    assert map.size() == 1;
    assert map.containsKey(17);
    // assert map.putIfAbsent(217, "E") == null;
    assert map.put(217, "E") == null;  // Was putIfAbsent
    assert map.get(217).equals("E");
    assert map.size() == 2;
    assert map.containsKey(217);
    // assert map.putIfAbsent(34, "F") == null;
    assert map.put(34, "F") == null;   // Was putIfAbsent
    map.forEach((k, v) -> System.out.printf("%10d maps to %s%n", k, v));
    assert map.size() == 3;
    assert map.get(17).equals("B") && map.containsKey(17);
    assert map.get(217).equals("E") && map.containsKey(217);
    assert map.get(34).equals("F") && map.containsKey(34);
    map.forEach((k, v) -> System.out.printf("%10d maps to %s%n", k, v));    
    assert map.size() == 3;
    assert map.get(17).equals("B") && map.containsKey(17);
    assert map.get(217).equals("E") && map.containsKey(217);
    assert map.get(34).equals("F") && map.containsKey(34);
    map.forEach((k, v) -> System.out.printf("%10d maps to %s%n", k, v));    
  }

  private static void testMapConcurrent(final int threadCount, int perThread, int range, 
					final OurMap<Integer, String> map) 
    throws Exception
  {
    System.out.printf("%nConcurrent test: %s%n", map.getClass());
    final CyclicBarrier barrier = new CyclicBarrier(threadCount + 1);
    final Thread[] threads = new Thread[threadCount];
    final AtomicLong keySumSum = new AtomicLong();
    final AtomicIntegerArray addedBySums = new AtomicIntegerArray(threadCount);
    for (int t=0; t<threadCount; t++) {
      final int myThread = t;
      threads[t] = new Thread(new Runnable() { public void run() {
	try { barrier.await(); }
	catch (Exception exn) { throw new RuntimeException(exn); }
        final Random random = new Random();
	// addedBy[u] = number of entries added by thread number u
	final int[] addedBy = new int[threadCount];
	// Sum of keys added, minus sum of keys removed, by this thread
	long keySum = 0;
        for (int i=0; i<perThread; i++) {
          final Integer key = random.nextInt(range);
	  final String value = String.format("%02d:%d", myThread, key);
	  // 	  System.out.print(value + " ");
          if (!map.containsKey(key)) {
            // Add key with probability 80%
            if (random.nextDouble() < 0.80) {
	      String oldValue = map.put(key, value);
              if (null != oldValue) {  // Already there, remove old value, add new
		int oldThread = Integer.parseInt(oldValue.substring(0,2));
		addedBy[oldThread]--;
		addedBy[myThread]++;
	      } else { 
		keySum += key; // Add to sum if new to the map
		addedBy[myThread]++;
	      }
	    }
          } else { // Key was there, remove with probability 30% then reinsert
	    String v = map.get(key);
	    if (v != null) {
	      final int valueKey = Integer.parseInt(v.substring(3));
	      if (key != valueKey)
		System.out.printf("ERROR: key = % d != valueKey = %d%n", key, valueKey);
	    }
            if (random.nextDouble() < 0.30) {
	      String oldValue = map.remove(key);
              if (null != oldValue) {
		keySum -= key; // Subtract from sum if removed
		int oldThread = Integer.parseInt(oldValue.substring(0,2));
		addedBy[oldThread]--;
	      }
              // if (null == map.putIfAbsent(key, value)) {
	      // 	keySum += key; // Add to sum if new to the map
	      // 	addedBy[myThread]++;
	      // }
            }
	  }
        }
        // System.out.printf("Thread %02d keySum = %d addedBy = %d%n", 
	// 		  myThread, keySum, addedBy[myThread]);
	keySumSum.getAndAdd(keySum);
	for (int u=0; u<threadCount; u++) 
	  addedBySums.getAndAdd(u, addedBy[u]);
	try { barrier.await(); }
	catch (Exception exn) { throw new RuntimeException(exn); }
      }});
    }
    for (int t=0; t<threadCount; t++) 
      threads[t].start();
    barrier.await();		// Start all threads at the same time 
    Thread.sleep(1); 
    barrier.await();		// Wait for all threads to complete
    final long[] actualKeySum = new long[1], 
      actualSize = new long[1];
    final int[] actualAddedBy = new int[threadCount];
    map.forEach(new BiConsumer<Integer,String>() { public void accept(Integer k, String v) {
      actualKeySum[0] += k;
      actualSize[0]++;
      final int madeByThread = Integer.parseInt(v.substring(0,2));
      final int valueKey = Integer.parseInt(v.substring(3));
      if (k != valueKey)
	System.out.printf("ERROR: k = % d != valueKey = %d%n", k, valueKey);
      actualAddedBy[madeByThread]++; 
    }});
    if (actualSize[0] != map.size())
      System.out.printf("ERROR: actualSize = %d != map.size() = %d%n", 
			actualSize[0], map.size());

    if (keySumSum.get() != actualKeySum[0])
      System.out.printf("ERROR: keySumSum.get() = %d != actualKeySum[0] = %d%n", 
			keySumSum.get(), actualKeySum[0]);
    for (int t=0; t<threadCount; t++) 
      if (addedBySums.get(t) != actualAddedBy[t])
	System.out.printf("ERROR: Thread %02d: addedBySums = %d != actualAddedBy = %d%n", 
			  t, addedBySums.get(t), actualAddedBy[t]);
  }
  
  public static void SystemInfo() {
    System.out.printf("# OS:   %s; %s; %s%n", 
                      System.getProperty("os.name"), 
                      System.getProperty("os.version"), 
                      System.getProperty("os.arch"));
    System.out.printf("# JVM:  %s; %s%n", 
                      System.getProperty("java.vendor"), 
                      System.getProperty("java.version"));
    // This line works only on MS Windows:
    System.out.printf("# CPU:  %s%n", System.getenv("PROCESSOR_IDENTIFIER"));
    java.util.Date now = new java.util.Date();
    System.out.printf("# Date: %s%n", 
      new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ").format(now));
  }
}
