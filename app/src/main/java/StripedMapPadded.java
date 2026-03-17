// Thread-safe synchronized hash map using lock striping
// sestoft@itu.dk * 2025-06-26

// Based on 2014 TestHashMapSolution.java

import java.util.function.BiConsumer;

// A hash map that permits thread-safe concurrent operations, using
// lock striping (intrinsic locks on Objects created for the purpose).

// The bucketCount must be a multiple of the number lockCount of
// stripes, so that h % lockCount == (h % bucketCount) % lockCount and
// so that h % lockCount is invariant under doubling the number of
// buckets in method reallocateBuckets.  Otherwise there is a risk of
// locking a stripe, only to have the relevant entry moved to a
// different stripe by an intervening call to reallocateBuckets.

// This implementation differs from StripedMap only by padding
// the locks and sizes arrays (putting dummy elements between any two
// useful elements) so as to avoid false sharing of cache lines.  This
// simple trick improves performance for large thread counts.

public class StripedMapPadded<K,V> implements OurMap<K,V> {
  // Synchronization policy: 
  //   buckets[hash] is guarded by locks[hash%lockCount]
  //   sizes[s]      is guarded by locks[s]
  private volatile ItemNode<K,V>[] buckets;
  private final int lockCount;
  private final Object[] locks;
  private final int[] sizes;
  private final static int padding = 16;

  public StripedMapPadded(int lockCount) {
    int bucketCount = lockCount; // Must be a multiple of lockCount
    this.lockCount = lockCount;
    this.buckets = makeBuckets(bucketCount);
    this.locks = new Object[lockCount * padding];
    this.sizes = new int[lockCount * padding];
    for (int s=0; s<lockCount * padding; s++) 
      this.locks[s] = new Object();
  }

  @SuppressWarnings("unchecked") 
  private static <K,V> ItemNode<K,V>[] makeBuckets(int size) {
    // Java's @$#@?!! type system requires this unsafe cast    
    return (ItemNode<K,V>[])new ItemNode[size];
  }

  // Protect against poor hash functions and make non-negative
  private static <K> int getHash(K k) {
    final int kh = k.hashCode();
    return (kh ^ (kh >>> 16)) & 0x7FFFFFFF;  
  }

  // Return true if key k is in map, else false
  public boolean containsKey(K k) {
    final int h = getHash(k), s = h % lockCount;
    synchronized (locks[s * padding]) {
      final int hash = h % buckets.length;
      return ItemNode.search(buckets[hash], k) != null;
    }
  }

  // Return value v associated with key k, or null
  public V get(K k) {
    final int h = getHash(k), s = h % lockCount;
    synchronized (locks[s * padding]) {
      final int hash = h % buckets.length;
      ItemNode<K,V> node = ItemNode.search(buckets[hash], k);
      if (node != null) 
        return node.v;
      else
        return null;
    }
  }

  public int size() {
    int result = 0;
    for (int s=0; s<lockCount; s++) 
      synchronized (locks[s * padding]) {
        result += sizes[s * padding];
      }
    return result;
  }

  // Put v at key k, or update if already present.  The logic here has
  // become more contorted because we must not hold the stripe lock
  // when calling reallocateBuckets, otherwise there will be deadlock
  // when two threads working on different stripes try to reallocate
  // at the same time.
  
  public V put(K k, V v) {
    final int h = getHash(k), s = h % lockCount;
    int afterSize = 0;
    V old = null;
    synchronized (locks[s * padding]) {
      final int hash = h % buckets.length;
      final ItemNode<K,V> node = ItemNode.search(buckets[hash], k);
      if (node != null) {
        old = node.v;
        node.v = v;
      } else {
        buckets[hash] = new ItemNode<K,V>(k, v, buckets[hash]);
        afterSize = ++sizes[s * padding];
      }
    }
    if (afterSize * lockCount > buckets.length)
      reallocateBuckets(buckets);
    return old;
  }

  // Remove and return the value at key k if any, else return null
  public V remove(K k) {
    final int h = getHash(k), s = h % lockCount;
    synchronized (locks[s * padding]) {
      final int hash = h % buckets.length;
      ItemNode<K,V> prev = buckets[hash];
      if (prev == null) 
        return null;
      else if (k.equals(prev.k)) {      // Delete first ItemNode
        V old = prev.v;
        sizes[s * padding]--; 
        buckets[hash] = prev.next;
        return old;
      } else {                          // Search later ItemNodes
        while (prev.next != null && !k.equals(prev.next.k))
          prev = prev.next;
        // Now prev.next == null || k.equals(prev.next.k)
        if (prev.next != null) {        // Delete ItemNode prev.next
          V old = prev.next.v;
          sizes[s * padding]--; 
          prev.next = prev.next.next;
          return old;
        } else
          return null;
      }
    }
  }

  // Iterate over the hashmap's entries one stripe at a time; less locking
  public void forEach(BiConsumer<K,V> consumer) {
    final ItemNode<K,V>[] bs = buckets;
    for (int s=0; s<lockCount; s++) 
      synchronized (locks[s * padding]) {
        for (int hash=s; hash<bs.length; hash+=lockCount) {
          ItemNode<K,V> node = bs[hash];
          while (node != null) {
            consumer.accept(node.k, node.v);
            node = node.next;
          }
        }
      }
  }

  // First lock all stripes.  Then double bucket table size, rehash,
  // and redistribute entries.  Since the number of stripes does not
  // change, and since buckets.length is a multiple of lockCount, a
  // key that belongs to stripe s because (getHash(k) % N) %
  // lockCount == s will continue to belong to stripe s.  Hence the
  // sizes array need not be recomputed.

  // In any case, do not reallocate if the buckets field was updated
  // since the need for reallocation was discovered. CAN THIS HAPPEN? 

  public void reallocateBuckets(final ItemNode<K,V>[] oldBuckets) {
    lockAllAndThen(new Runnable() { 
        public void run() {
	  final ItemNode<K,V>[] bs = buckets;
	  if (oldBuckets == bs) {
	    // System.out.printf("Reallocating from %d buckets%n", buckets.length);
	    final ItemNode<K,V>[] newBuckets = makeBuckets(2 * bs.length);
	    for (int hash=0; hash<bs.length; hash++) {
	      ItemNode<K,V> node = bs[hash];
	      while (node != null) {
		final int newHash = getHash(node.k) % newBuckets.length;
		ItemNode<K,V> next = node.next;
		node.next = newBuckets[newHash];
		newBuckets[newHash] = node;
		node = next;
	      }
	    }
	    buckets = newBuckets;
	  }
	}
      });
  }
  
  // Lock all stripes, perform the action, then unlock all stripes
  private void lockAllAndThen(Runnable action) {
    lockAllAndThen(0, action);
  }

  private void lockAllAndThen(int nextStripe, Runnable action) {
    if (nextStripe >= lockCount)
      action.run();
    else 
      synchronized (locks[nextStripe * padding]) {
        lockAllAndThen(nextStripe + 1, action);
      }
  }

  static class ItemNode<K,V> {
    private final K k;
    private V v;
    private ItemNode<K,V> next;
    
    public ItemNode(K k, V v, ItemNode<K,V> next) {
      this.k = k;
      this.v = v;
      this.next = next;
    }

    // Assumes locks[getHash(k) % lockCount] is held by the thread
    public static <K,V> ItemNode<K,V> search(ItemNode<K,V> node, K k) {
      while (node != null && !k.equals(node.k))
        node = node.next;
      return node;
    }
  }
}
