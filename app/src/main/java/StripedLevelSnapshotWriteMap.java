// Thread-safe striped level (or 2D) write hash map with snapshot, experimental
// sestoft@itu.dk * 2025-05-26

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

// A hashmap that permits thread-safe concurrent operations, using
// lock striping (intrinsic lock on Stripe objects), in a two-level
// format where each stripe has a separate array of buckets. It has
// immutable ItemNodes, so that reads do not need to lock at all, only
// need visibility of writes, which is ensured through the stripe's
// AtomicInteger field called size. 

// Let S=lockCount be the number of stripes and logS be the base 2
// logarithm of S. Then we effectively use the lower logS bits of the
// hash code to determine the stripe, and the remaining (32 - logS)
// upper bits (modulo the stripe bucket array length) to determine the
// index within the stripe's bucket array.

// The stripes may have different bucket array lengths, and may be
// reallocated separately, taking only the stripe's own lock, thereby
// increasing concurrency.

// Each stripe is represented by a Stripe<K,V> object, which holds the
// stripe's size (number of map entries) and a reference to the
// stripe's bucket array. The Stripe object's intrinsic lock is used
// to protect operations on the bucket array.

// It is unlikely that two Stripe objects are on the same cache line,
// but a Stripe object and either its size object or its buckets array
// may be.  Hence might be worth padding the Stripe object with dummy fields. 

// Since the bucket lists are immutable, a snapshot can be created by
// copying each Stripe object, with its bucket array and size field,
// one by one and creating a top-level array with references to the
// new Stripe objects.  It is necessary to take each old stripe's lock
// before copying, or else the old stripe's bucket array contents and
// size field may be updated during copying. 

public class StripedLevelSnapshotWriteMap<K,V> implements OurMap<K,V> {
  // Synchronization policy: writing to
  //   stripe.buckets[index] is guarded by intrinsic lock on stripe
  // Visibility of writes to reads is ensured by writes writing to
  // the stripe's size component (even if size does not change) and
  // reads reading from the stripe's size component. 
  private final Stripe<K,V>[] stripes;
  private final int lockCount;

  private static class Stripe<K,V> {
    private volatile ItemNode<K,V>[] buckets;
    private final AtomicInteger size;

    public Stripe(int stripeBucketCount) {
      this.buckets = makeBuckets(stripeBucketCount);
      this.size = new AtomicInteger();
    }
  }
  
  public StripedLevelSnapshotWriteMap(int lockCount) {
    int bucketCount = lockCount; // Must be a multiple of lockCount
    this.lockCount = lockCount;
    this.stripes = makeStripeArray(lockCount);
    for (int s=0; s<lockCount; s++) 
      this.stripes[s] = new Stripe<K,V>(bucketCount / lockCount);
  }

  private StripedLevelSnapshotWriteMap(Stripe<K,V>[] stripes, int lockCount) {
    this.stripes = stripes;
    this.lockCount = lockCount;
  }
  
  @SuppressWarnings("unchecked") 
  private static <K,V> ItemNode<K,V>[] makeBuckets(int size) {
    // Java's @$#@?!! type system requires "unsafe" cast here:
    return (ItemNode<K,V>[])new ItemNode[size];
  }

  @SuppressWarnings("unchecked") 
  private static <K,V> Stripe<K,V>[] makeStripeArray(int size) {
    // Java's @$#@?!! type system requires "unsafe" cast here:
    return (Stripe<K,V>[])new Stripe[size];
  }
  
  // Protect against poor hash functions and make non-negative
  private static <K> int getHash(K k) {
    final int kh = k.hashCode();
    return (kh ^ (kh >>> 16)) & 0x7FFFFFFF;  
  }

  // Return true if key k is in map, else false
  public boolean containsKey(K k) {
    final int h = getHash(k), s = h % lockCount, rest = h / lockCount;
    Stripe<K,V> stripe = stripes[s];
    final ItemNode<K,V>[] bs = stripe.buckets;
    // The sizes access is necessary for visibility of bs elements
    return stripe.size.get() != 0 && ItemNode.search(bs[rest % bs.length], k, null);
  }

  // Return value v associated with key k, or null
  public V get(K k) {
    final int h = getHash(k), s = h % lockCount, rest = h / lockCount;
    Stripe<K,V> stripe = stripes[s];
    final ItemNode<K,V>[] bs = stripe.buckets;
    Holder<V> value = new Holder<V>();
    // The sizes access is necessary for visibility of bs elements
    if (stripe.size.get() != 0 && ItemNode.search(bs[rest % bs.length], k, value))
      return value.get();
    else
      return null;
  }

  public int size() {
    int result = 0;
    for (int s=0; s<lockCount; s++) 
      result += stripes[s].size.get();
    return result;
  }

  // Put v at key k, or update if already present.  Must hold the
  // stripe's lock while calling reallocateBucket.
  public V put(K k, V v) {
    final int h = getHash(k), s = h % lockCount, rest = h / lockCount;
    Stripe<K,V> stripe = stripes[s];
    synchronized (stripe) {
      final ItemNode<K,V>[] bs = stripe.buckets;
      final Holder<V> old = new Holder<V>();
      final int index = rest % bs.length;
      final ItemNode<K,V> node = bs[index], 
        newNode = ItemNode.delete(node, k, old);
      bs[index] = new ItemNode<K,V>(k, v, newNode);
      // Write for visibility; increment if k was not already in map
      int afterSize = stripe.size.addAndGet(newNode == node ? 1 : 0);
      if (afterSize > bs.length) 
	reallocateBucket(stripe, bs);
      return old.get();
    }
  }

  // Remove and return the value at key k if any, else return null
  public V remove(K k) {
    final int h = getHash(k), s = h % lockCount, rest = h / lockCount;
    Stripe<K,V> stripe = stripes[s];
    synchronized (stripe) {
      final ItemNode<K,V>[] bs = stripe.buckets;
      final Holder<V> old = new Holder<V>();
      final int index = rest % bs.length;
      final ItemNode<K,V> node = bs[index], 
        newNode = ItemNode.delete(node, k, old);
      if (newNode != node) { // Removed something from node list, so update
        bs[index] = newNode;
        stripe.size.getAndDecrement();   // Visibility
      } 
      return old.get();
    }
  }

  // Iterate over the hashmap's entries one stripe at a time.  
  public void forEach(BiConsumer<K,V> consumer) {
    for (Stripe<K,V> stripe : stripes) {
      if (stripe.size.get() != 0) { // Visibility
	final ItemNode<K,V>[] bs = stripe.buckets;
        for (ItemNode<K,V> node : bs) {
          while (node != null) {
            consumer.accept(node.k, node.v);
            node = node.next;
          }
        }
      }
    }
  }

  // Method reallocateBucket(stripe, oldBuckets) must be called while
  // holding the intrinsic lock on stripe.  Double the bucket table
  // size, rehash, and redistribute entries based on the upper bits of
  // the hash value. Obviously the stripe's size does not change.

  // Unlike StripedWriteMap, there is no need to check whether another
  // thread reallocated in the meantime, since no other thread can add
  // to the stripe between this thread discovering the need for
  // reallocation and this threads performing the reallocation.

  public void reallocateBucket(Stripe<K,V> stripe, final ItemNode<K,V>[] oldBuckets) {
    final ItemNode<K,V>[] bs = stripe.buckets;
    // System.out.printf("Reallocating from %d buckets%n", buckets.length);
    final ItemNode<K,V>[] newBuckets = makeBuckets(2 * bs.length);
    for (int hash=0; hash<bs.length; hash++) {
      ItemNode<K,V> node = bs[hash];
      while (node != null) {
	final int newHash = (getHash(node.k) / lockCount) % newBuckets.length;
	final ItemNode<K,V> tail = newBuckets[newHash];
	// Small optimization: reuse nodes if possible
	newBuckets[newHash] 
	  = node.next==tail ? node : new ItemNode<K,V>(node.k, node.v, tail);
	node = node.next;
      }
      stripe.buckets = newBuckets; // Visibility: buckets field is volatile
    }
  }

  // Experimental, untested 2025-05-26
  
  public StripedLevelSnapshotWriteMap<K,V> snapshot() {
    final Stripe<K,V>[] newStripes = makeStripeArray(lockCount);
    for (int s=0; s<lockCount; s++) {
      Stripe<K,V> oldStripe = stripes[s];
      synchronized (oldStripe) {
	Stripe<K,V> newStripe = new Stripe<K,V>(oldStripe.buckets.length);
	for (int index=0; index<oldStripe.buckets.length; index++)
	  newStripe.buckets[index] = oldStripe.buckets[index];
	newStripe.size.set(oldStripe.size.get());
	newStripes[s] = newStripe;
      }
    }
    return new StripedLevelSnapshotWriteMap<K,V>(newStripes, lockCount);
  }
  
  static class ItemNode<K,V> {
    private final K k;
    private final V v;
    private final ItemNode<K,V> next;
    
    public ItemNode(K k, V v, ItemNode<K,V> next) {
      this.k = k;
      this.v = v;
      this.next = next;
    }

    // These work on immutable data only, no synchronization needed.

    public static <K,V> boolean search(ItemNode<K,V> node, K k, Holder<V> old) {
      while (node != null) 
        if (k.equals(node.k)) {
          if (old != null) 
            old.set(node.v);
          return true;
        } else 
          node = node.next;
      return false;
    }
    
    public static <K,V> ItemNode<K,V> delete(ItemNode<K,V> node, K k, Holder<V> old) {
      if (node == null) 
        return null; 
      else if (k.equals(node.k)) {
        old.set(node.v);
        return node.next;
      } else {
        final ItemNode<K,V> newNode = delete(node.next, k, old);
        if (newNode == node.next) 
          return node;
        else 
          return new ItemNode<K,V>(node.k, node.v, newNode);
      }
    }
  }
  
  // Object to hold a "by reference" parameter.  For use only on a
  // single thread, so no need for "volatile" or synchronization.

  static class Holder<V> {
    private V value;
    public V get() { 
      return value; 
    }
    public void set(V value) { 
      this.value = value;
    }
  }
}
