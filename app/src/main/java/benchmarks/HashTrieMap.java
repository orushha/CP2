// HashTrieMap, lock-free concurrent hash tries, design by Prokopec,
// Bagwell, Odersky 2011-2017.

// This implementation is from ConcurrentHashTrieMap by Florent Le
// Gall 2011, obtained from
// https://github.com/flegall/concurrent-hash-trie on 2025-05-22 It
// gives credit for idea and implementation techniques to Aleksandar
// Prokopec:
// http://infoscience.epfl.ch/record/166908/files/ctries-techreport.pdf

// In this version, bulk operators, iterators, Java interfaces etc
// have been removed for simplicity and clarity, and implementation of
// the OurMap<K,V> interface has been added.
package benchmarks;

import java.util.function.BiConsumer;
import java.util.concurrent.atomic.AtomicReferenceFieldUpdater;

public class HashTrieMap<K, V> implements OurMap<K,V> {
  private final INode root;
  private final byte width; // in bits

  public HashTrieMap () {
    this (6);
  }
  
  /**
   * Builds a {@link HashTrieMap} instance
   * 
   * @param width
   *            the Trie width in power-of-two exponents. Values are expected
   *            between between 1 & 6, other values will be clamped.
   *            <p>
   *            The width defines the "speed" of the trie:
   *            <ul>
   *            <li>A value of 1: gives an actual width of two items per
   *            level, hence the trie is O(Log2(N))</li>
   *            <li>A value of 6: gives an actual width of 64 items per level,
   *            hence the trie is O(Log64(N)</li>
   *            </ul>
   */
  public HashTrieMap(final int width) {
    this.root = new INode (new CNode<K, V> ());
    this.width = (byte) (Math.min(6, Math.max(1, width)));
  }

  public boolean containsKey (final K key) {
    return null != lookup(key);
  }
    
  public V get (final K key) {
    return lookup (key);
  }

  public V put (final K key, final V value) {
    return insert (key, value, new Constraint<V> (ConstraintType.NONE, null));
  }

  public V remove (final K key) {
    return delete (key, HashTrieMap.<V>noConstraint ());
  }

  public void forEach(BiConsumer<K,V> consumer) {
    SNode<K, V> nextSNode = lookupNext(null);
    while (nextSNode != null) {
      KeyValueNode<K, V> nextKV = nextSNode.next(null);
      while (nextKV != null) {
	consumer.accept(nextKV.key, nextKV.value);
	nextKV = nextSNode.next(nextKV);
      }
      nextSNode = lookupNext(nextSNode);
    }
  }

  public int size() {
    final int[] count = new int[1];
    forEach((k, v) -> { count[0]++; });
    return count[0];
  }
  
  // Inserts or updates a key/value mapping.
  V insert (final K key, final V value, final Constraint<V> constraint) {
    notNullKey (key);
    notNullValue (value);
    final int hc = hash (key);
    while (true) {
      final Result<V> res = iinsert (this.root, hc, key, value, 0, null, constraint);
      switch (res.type) {
      case FOUND:
	return res.result;
      case NOTFOUND:
	return null;
      case RESTART:
	continue;
      case REJECTED:
	if (ConstraintType.PUT_IF_ABSENT == constraint.type) {
	  return res.result;
	} else if (ConstraintType.REPLACE_IF_MAPPED == constraint.type) {
	  return null;
	} else if (ConstraintType.REPLACE_IF_MAPPED_TO == constraint.type) {
	  return res.result;
	} else {
	  throw new RuntimeException ("Unexpected case: " + constraint.type);
	}
      default:
	throw new RuntimeException ("Unexpected case: " + res.type);
      }
    }
  }

  // Looks up the value associated to a key
  V lookup (final K key) {
    notNullKey (key);
    final int hc = hash (key);
    while (true) {
      // Getting lookup result
      final Result<V> res = ilookup (this.root, hc, key, 0, null);
      switch (res.type) {
      case FOUND:
	return res.result;
      case NOTFOUND:
	return null;
      case RESTART:
	continue;
      default:
	throw new RuntimeException ("Unexpected case: " + res.type);
      }
    }
  }

  // Removes a key/value mapping, returns the removed value, null otherwise
  V delete (final K key, final Constraint<V> constraint) {
    notNullKey (key);
    final int hc = hash (key);
    while (true) {
      // Getting remove result
      final Result<V> res = idelete (this.root, hc, key, 0, null, constraint);
      switch (res.type) {
      case FOUND:
	return res.result;
      case NOTFOUND:
	return null;
      case RESTART:
	continue;
      case REJECTED:
	if (ConstraintType.REMOVE_IF_MAPPED_TO == constraint.type) {
	  return res.result;
	} else {
	  throw new RuntimeException ("Unexpected case: " + constraint.type);
	}
      default:
	throw new RuntimeException ("Unexpected case: " + res.type);
      }
    }
  }
    
  SNode<K, V> lookupNext (final SNode<K, V> current) {
    if (current != null) {
      final int hc = current.hash ();
      while (true) {
	// Getting lookup result
	final Result<SNode<K, V>> res = ilookupNext (this.root, hc, 0, null);
	switch (res.type) {
	case FOUND:
	  return res.result;
	case NOTFOUND:
	  return null;
	case RESTART:
	  continue;
	default:
	  throw new RuntimeException ("Unexpected case: " + res.type);
	}
      }
    } else {
      while (true) {
	// Getting lookup result
	final Result<SNode<K, V>> res = ilookupFirst (this.root, 0, null);
	switch (res.type) {
	case FOUND:
	  return res.result;
	case NOTFOUND:
	  return null;
	case RESTART:
	  continue;
	default:
	  throw new RuntimeException ("Unexpected case: " + res.type);
	}
      }
    }
  }
  
  private Result<V> ilookup (final INode i, final int hashcode, final K k,
			     final int level, final INode parent) {
    final MainNode main = i.getMain ();
    // Usual case
    if (main instanceof CNode) {
      @SuppressWarnings("unchecked")
	final CNode<K, V> cn = (CNode<K, V>) main;
      final FlagPos flagPos = flagPos (hashcode, level, cn.bitmap, this.width);
      
      // Asked for a hash not in trie
      if (0L == (flagPos.flag & cn.bitmap)) {
	return new Result<V> (ResultType.NOTFOUND, null);
      }
      
      final BranchNode an = cn.array [flagPos.position];
      if (an instanceof INode) {
	// Looking down
	final INode sin = (INode) an;
	return ilookup (sin, hashcode, k, level + this.width, i);
      }
      if (an instanceof SNode) {
	// Found the hash locally, let's see if it matches
	@SuppressWarnings("unchecked")
	  final SNode<K, V> sn = (SNode<K, V>) an;
	if (sn.hash () == hashcode) {
	  final V v = sn.get (k);
	  if (null != v) {
	    return new Result<V> (ResultType.FOUND, v);
	  } else {
	    return new Result<V> (ResultType.NOTFOUND, null);
	  }
	} else {
	  return new Result<V> (ResultType.NOTFOUND, null);
	}
      }
    }

    // Cleaning up trie
    if (main instanceof TNode) {
      clean (parent, level - this.width);
      return new Result<V> (ResultType.RESTART, null);
    }
    throw new RuntimeException ("Unexpected case: " + main);
  }
  
  private Result<V> iinsert (final INode i, final int hashcode, final K k, final V v,
			     final int level, final INode parent, final Constraint<V> constraint) {
    final MainNode main = i.getMain ();    
    // Usual case
    if (main instanceof CNode) {
      @SuppressWarnings("unchecked")
	final CNode<K, V> cn = (CNode<K, V>) main;
      final FlagPos flagPos = flagPos (hashcode, level, cn.bitmap, this.width);
      
      // Asked for a hash not in trie, let's insert it
      if (0L == (flagPos.flag & cn.bitmap)) { 
	
	// Check constraints
	if (ConstraintType.REPLACE_IF_MAPPED_TO == constraint.type ||
	    ConstraintType.REPLACE_IF_MAPPED == constraint.type) {
	  return new Result<V> (ResultType.REJECTED, null);
	}
        
	final SNode<K, V> snode = new SingletonSNode<K, V> (k, v);
	final CNode<K, V> ncn = cn.inserted (flagPos, snode);
	if (i.casMain (main, ncn)) {
	  return new Result<V> (ResultType.FOUND, null);
	} else {
	  return new Result<V> (ResultType.RESTART, null);
	}
      }
      
      final BranchNode an = cn.array [flagPos.position];
      if (an instanceof INode) {
	// Looking down
	final INode sin = (INode) an;
	return iinsert (sin, hashcode, k, v, level + this.width, i, constraint);
      }
      if (an instanceof SNode) {
	@SuppressWarnings("unchecked")
	  final SNode<K, V> sn = (SNode<K, V>) an;
	
	// Found the hash locally, let's see if it matches
	if (sn.hash () == hashcode) {
	  final V previousValue = sn.get (k);
          
	  // Check constraints
	  if (ConstraintType.PUT_IF_ABSENT == constraint.type && null != previousValue) {
	    return new Result<V> (ResultType.REJECTED, previousValue);
	  }
	  if (ConstraintType.REPLACE_IF_MAPPED_TO == constraint.type && 
	      !previousValue.equals (constraint.to)) {
	    return new Result<V> (ResultType.REJECTED, previousValue);
	  }
                    
	  final SNode<K, V> nsn = sn.put (k, v);
	  final CNode<K, V> ncn = cn.updated (flagPos.position, nsn);
	  if (i.casMain (main, ncn)) {
	    return new Result<V> (ResultType.FOUND, previousValue);
	  } else {
	    return new Result<V> (ResultType.RESTART, null);
	  }
	} else {
	  // Check constraints
	  if (ConstraintType.REPLACE_IF_MAPPED_TO == constraint.type ||
	      ConstraintType.REPLACE_IF_MAPPED == constraint.type) {
	    return new Result<V> (ResultType.REJECTED, null);
	  }
          
	  final SNode<K, V> nsn = new SingletonSNode<K, V> (k, v);
	  // Creates a sub-level
	  final CNode<K, V> scn = new CNode<K, V> (sn, nsn, level + this.width, this.width);
	  final INode nin = new INode (scn);
	  final CNode<K, V> ncn = cn.updated (flagPos.position, nin);
	  if (i.casMain (main, ncn)) {
	    return new Result<V> (ResultType.FOUND, null);
	  } else {
	    return new Result<V> (ResultType.RESTART, null);
	  }
	}
      }
    }
    
    // Cleaning up trie
    if (main instanceof TNode) {
      clean (parent, level - this.width);
      return new Result<V> (ResultType.RESTART, null);
    }
    throw new RuntimeException ("Unexpected case: " + main);
  }

  private Result<V> idelete (final INode i, final int hashcode, final K k,
			     final int level, final INode parent, final Constraint<V> constraint) {
    final MainNode main = i.getMain ();

    // Usual case
    if (main instanceof CNode) {
      @SuppressWarnings("unchecked")
	final CNode<K, V> cn = (CNode<K, V>) main;
      final FlagPos flagPos = flagPos (hashcode, level, cn.bitmap, this.width);
      
      // Asked for a hash not in trie
      if (0L == (flagPos.flag & cn.bitmap)) {
	return new Result<V> (ResultType.NOTFOUND, null);
      }
      
      Result<V> res = null;
      final BranchNode an = cn.array [flagPos.position];
      if (an instanceof INode) {
	// Looking down
	final INode sin = (INode) an;
	res = idelete (sin, hashcode, k, level + this.width, i, constraint);
      }
      if (an instanceof SNode) {
	// Found the hash locally, let's see if it matches
	@SuppressWarnings("unchecked")
	  final SNode<K, V> sn = (SNode<K, V>) an;
	if (sn.hash () == hashcode) {
	  final V previous = sn.get (k);
	  // Checking constraint first
	  if (null == previous) {
	    res = new Result<V> (ResultType.NOTFOUND, null);
	  } else if (ConstraintType.REMOVE_IF_MAPPED_TO == constraint.type &&
		     !constraint.to.equals (previous)) {
	    res = new Result<V> (ResultType.REJECTED, previous);
	  } else {
	    final SNode<K, V> nsn = sn.removed (k);
	    final MainNode replacement;
	    if (null != nsn) {
	      replacement = cn.updated (flagPos.position, nsn);
	    } else {
	      final CNode<K, V> ncn = cn.removed (flagPos);
	      replacement = toContracted (ncn, level);
	    }
	    if (i.casMain (main, replacement)) {
	      res = new Result<V> (ResultType.FOUND, previous);
	    } else {
	      res = new Result<V> (ResultType.RESTART, null);
	    }
	  }
	} else {
	  res = new Result<V> (ResultType.NOTFOUND, null);
	}
      }
      if (null == res) {
	throw new RuntimeException ("Unexpected case: " + an);
      }
      if (res.type == ResultType.NOTFOUND || res.type == ResultType.RESTART) {
	return res;
      }
      
      if (i.getMain () instanceof TNode) {
	cleanParent (parent, i, hashcode, level - this.width);
      }
      return res;
    }
    
    // Cleaning up trie
    if (main instanceof TNode) {
      clean (parent, level - this.width);
      return new Result<V> (ResultType.RESTART, null);
    }
    throw new RuntimeException ("Unexpected case: " + main);
  }
  
  private Result<SNode<K, V>> ilookupFirst (final INode i, final int level, final INode parent) {
    final MainNode main = i.getMain ();

    // Usual case
    if (main instanceof CNode) {
      @SuppressWarnings("unchecked")
	final CNode<K, V> cn = (CNode<K, V>) main;
      if (cn.bitmap == 0L) {
	return new Result<SNode<K, V>> (ResultType.NOTFOUND, null);
      } else {
	return ipickupFirst (cn.array [0], level, i);
      }
    }
    
    // Cleaning up trie
    if (main instanceof TNode) {
      clean (parent, level - this.width);
      return new Result<SNode<K, V>> (ResultType.RESTART, null);
    }
    throw new RuntimeException ("Unexpected case: " + main);
  }

  private Result<SNode<K, V>> ilookupNext (final INode i, final int hashcode, 
					   final int level, final INode parent) {
    final MainNode main = i.getMain ();
    
    // Usual case
    if (main instanceof CNode) {
      @SuppressWarnings("unchecked")
	final CNode<K, V> cn = (CNode<K, V>) main;
      final FlagPos flagPos = flagPos (hashcode, level, cn.bitmap, this.width);
      
      // Asked for a hash not in trie
      if (0L == (flagPos.flag & cn.bitmap)) {
	return ipickupFirstSibling (cn, flagPos, 0, level, i);
      }
      
      final BranchNode an = cn.array [flagPos.position];
      if (an instanceof INode) {
	// Looking down
	final INode sin = (INode) an;
	final Result<SNode<K, V>> next = ilookupNext (sin, hashcode, level + this.width, i);
	switch (next.type) {
	case FOUND:
	  return next;
	case NOTFOUND:
	  return ipickupFirstSibling (cn, flagPos, 1, level, i);
	case RESTART:
	  return next;
	default:
	  throw new RuntimeException ("Unexpected case: " + next.type);
	}
      }
      if (an instanceof SNode) {
	@SuppressWarnings("unchecked")
	  final SNode<K, V> sn = (SNode<K, V>) an;
	if (hashcode + Integer.MIN_VALUE >= sn.hash () + Integer.MIN_VALUE) {
	  return ipickupFirstSibling (cn, flagPos, 1, level, i);
	} else {
	  return new Result<SNode<K, V>> (ResultType.FOUND, sn);
	}
      }
    }
    
    // Cleaning up trie
    if (main instanceof TNode) {
      clean (parent, level - this.width);
      return new Result<SNode<K, V>> (ResultType.RESTART, null);
    }
    throw new RuntimeException ("Unexpected case: " + main);
  }
  
  private Result<SNode<K, V>> ipickupFirstSibling (final CNode<K, V> cn, 
            final FlagPos flagPos, 
            final int offset, 
            final int level, 
            final INode parent) {
        
    // Go directly to the next entry in the current node if possible
    if (flagPos.position + offset < cn.array.length) {
      final BranchNode an = cn.array [flagPos.position + offset];
      return ipickupFirst (an, level, parent);
    } else {
      return new Result<SNode<K, V>> (ResultType.NOTFOUND, null);
    }
  }

  private Result<SNode<K, V>> ipickupFirst (final BranchNode bn, final int level, final INode parent) {
    if (bn instanceof INode) {
      // Looking down
      final INode sin = (INode) bn;
      return ilookupFirst (sin, level + this.width, parent);
    }
    if (bn instanceof SNode) {
      // Found the SNode
      @SuppressWarnings("unchecked")
	final SNode<K, V> sn = (SNode<K, V>) bn;
      return new Result<SNode<K, V>> (ResultType.FOUND, sn);
    }
    throw new RuntimeException ("Unexpected case: " + bn);
  }
  
  private void cleanParent (final INode parent, final INode i, final int hashCode, final int level) {
    while (true) {
      final MainNode m = i.getMain ();
      final MainNode pm = parent.getMain ();
      if (pm instanceof CNode) {
	@SuppressWarnings("unchecked")
	  final CNode<K, V> pcn = (CNode<K, V>) pm;
	final FlagPos flagPos = flagPos (hashCode, level, pcn.bitmap, this.width);
	if (0L == (flagPos.flag & pcn.bitmap)) {
	  return;
	}
	final BranchNode sub = pcn.array [flagPos.position];
	if (sub != i) {
	  return;
	}
	if (m instanceof TNode) {
	  @SuppressWarnings("unchecked")
	    final SNode<K, V> untombed = ((TNode<K, V>) m).untombed ();
	  final CNode<K, V> ncn = pcn.updated (flagPos.position, untombed);
	  if (parent.casMain (pcn, toContracted (ncn, level))) {
	    return;
	  } else {
	    continue;
	  }
	}
      } else {
	return;
      }
    }
  }
  
  private void clean (final INode i, final int level) {
    final MainNode m = i.getMain ();
    if (m instanceof CNode) {
      @SuppressWarnings("unchecked")
	final CNode<K, V> cn = (CNode<K, V>) m;
      i.casMain (m, toCompressed (cn, level));
    }
  }
  
  private MainNode toCompressed (final CNode<K, V> cn, final int level) {
    final CNode<K, V> ncn = cn.copied ();
    
    // Resurrect tombed nodes.
    for (int i = 0; i < ncn.array.length; i++) {
      final BranchNode an = ncn.array [i];
      final TNode<K, V> tn = getTombNode (an);
      if (null != tn) {
	ncn.array [i] = tn.untombed ();
      }
    }
    
    return toContracted (ncn, level);
  }
  
  private MainNode toContracted (final CNode<K, V> cn, final int level) {
    if (level > 0 && 1 == cn.array.length) {
      final BranchNode bn = cn.array [0];
      if (bn instanceof SNode) {
	@SuppressWarnings("unchecked")
	  final SNode<K, V> sn = (SNode<K, V>) bn;
	return sn.tombed ();
      }
    }
    return cn;
  }
  
  private TNode<K, V> getTombNode (final BranchNode an) {
    if (an instanceof INode) {
      final INode in = (INode) an;
      final MainNode mn = in.getMain ();
      if (mn instanceof TNode) {
	@SuppressWarnings("unchecked")
	  final TNode<K, V> tn = (TNode<K, V>) mn;
	return tn;
      }
    }
    return null;
  }
  
  private void notNullValue (final V value) {
    if (value == null) {
      throw new NullPointerException ("The value must be non-null");
    }
  }
  
  private void notNullKey (final K key) {
    if (key == null) {
      throw new NullPointerException ("The key must be non-null");
    }
  }
  
  @SuppressWarnings("unchecked")
  static <V> Constraint<V> noConstraint () {
    return (Constraint<V>) NO_CONSTRAINT;
  }
  
  static int hash (final Object key) {
    int h = key.hashCode ();
    // This function ensures that hashCodes that differ only by
    // constant multiples at each bit position have a bounded
    // number of collisions (approximately 8 at default load factor).
    h ^= h >>> 20 ^ h >>> 12;
    return h ^ h >>> 7 ^ h >>> 4;
  }
  
  // Returns a copy an array with an updated value at given position
  static <T> T[] updated (final T[] src, final T[] dst, final T t, final int position) {
    System.arraycopy (src, 0, dst, 0, src.length);
    dst [position] = t;
    return dst;
  }

  // Returns a copy a BranchNode array with an inserted BranchNode value at given position
  static <T> T[] inserted (final T[] src, final T[] dst, final T t, final int position) {
    System.arraycopy (src, 0, dst, 0, position);
    System.arraycopy (src, position, dst, position + 1, src.length - position);
    dst [position] = t;
    return dst;
  }
  
  // Returns a copy of an array with a removed value at given position.
  static <T> T[] removed (final T[] src, final T[] dst, final int position) {
    System.arraycopy (src, 0, dst, 0, position);
    System.arraycopy (src, position + 1, dst, position, src.length - position - 1);
    return dst;
  }
  
  /**
   * Gets the flag value and insert position for an hashcode, level & bitmap.
   * 
   * @param hc
   *            the hashcode value
   * @param level
   *            the level (in bit progression)
   * @param bitmap
   *            the current {@link CNode}'s bitmap.
   * @param w
   *            the fan width (in bits)
   * @return a {@link FlagPos}'s instance for the specified hashcode, level &
   *         bitmap.
   */
  static FlagPos flagPos (final int hc, final int level, final long bitmap, final int w) {
    final long flag = flag (hc, level, w);
    final int pos = Long.bitCount (flag - 1 & bitmap);
    return new FlagPos (flag, pos);
  }

  /**
   * Gets the flag value for an hashcode level.
   * 
   * @param hc
   *            the hashcode value
   * @param level
   *            the level (in bit progression)
   * @param the
   *            fan width (in bits)
   * @return the flag value
   */
  static long flag (final int hc, final int level, final int w) {
    final int bitsRemaining = Math.min (w, 32 - level);
    final int subHash = hc >> level & (1 << bitsRemaining) - 1;
    final long flag = 1L << subHash;
    return flag;
  }
    
  static enum ConstraintType {
    NONE, 
    PUT_IF_ABSENT, 
    REPLACE_IF_MAPPED_TO,
    REPLACE_IF_MAPPED,
    REMOVE_IF_MAPPED_TO, 
  }
    
  static class Constraint<V> {
    public Constraint (final ConstraintType type, final V to) {
      this.type = type;
      this.to = to;
    }

    public final ConstraintType type; 
    public final V to;
  }

  static enum ResultType {
    FOUND, 
    NOTFOUND,
    REJECTED,
    RESTART
  }

  static class Result<V> {
    public Result (final ResultType type, final V result) {
      this.type = type;
      this.result = result;
    }
    
    public final V result;
    public final ResultType type;
  }

  /**
   * A Marker interface for what can be in an INode (CNode or SNode)
   */
  static interface MainNode {
  }

  /**
   * A Marker interface for what can be in a CNode array. (INode or SNode)
   */
  static interface BranchNode {
  }

  /**
   * A single node in the trie, why may contain several objects who share the
   * same hashcode.
   */
  static interface SNode<K, V> extends BranchNode {
    int hash ();
    V get (K k);
    SNode<K, V> put (K k, V v);
    // Return the copy of this SNode with the updated removal
    SNode<K, V> removed (K k);
    // Return a copied TNode for this instance.
    TNode<K, V> tombed ();
    // Get the next KeyValueNode instance following the current one, or the first if null
    KeyValueNode<K, V> next (KeyValueNode<K, V> current);
  }

  static interface TNode<K, V> extends MainNode {
    SNode<K, V> untombed ();
  }

  // A CAS-able Node which may reference either a CNode or and SNode
  static class INode implements BranchNode {
    public INode (final MainNode n) {
      INODE_UPDATER.set (this, n);
    }

    public MainNode getMain () {
      return this.main;
    }

    public boolean casMain (final MainNode expected, final MainNode update) {
      return INODE_UPDATER.compareAndSet (this, expected, update);
    }

    private static final AtomicReferenceFieldUpdater<INode, MainNode> INODE_UPDATER = 
      AtomicReferenceFieldUpdater.newUpdater (INode.class, MainNode.class, "main");
    
    private volatile MainNode main;
  }

  // A Node that may contain sub-nodes.
  static class CNode<K, V> implements MainNode {
    /**
     * Builds a copy of this {@link CNode} instance where a sub-node
     * designated by a position has been added .
     * 
     * @param flagPos
     *            a {@link FlagPos} instance
     * @param snode
     *            a {@link SNode} instance
     * @return a copy of this {@link CNode} instance with the inserted node.
     */
    public CNode<K, V> inserted (final FlagPos flagPos, final SNode<K, V> snode) {
      final BranchNode[] narr = HashTrieMap.inserted (this.array, 
                    new BranchNode [this.array.length + 1], 
                    snode, 
                    flagPos.position);
      return new CNode<K, V> (narr, flagPos.flag | this.bitmap);
    }

    /**
     * Builds a copy of this {@link CNode} instance where a sub
     * {@link BranchNode} designated by a position has been replaced by
     * another one.
     * 
     * @param position
     *            an integer position
     * @param bn
     *            a {@link BranchNode} instance
     * @return a copy of this {@link CNode} instance with the updated node.
     */
    public CNode<K, V> updated (final int position, final BranchNode bn) {
      final BranchNode[] narr = HashTrieMap.updated (this.array, 
                    new BranchNode [this.array.length], 
                    bn, 
                    position);
      return new CNode<K, V> (narr, this.bitmap);
    }

    /**
     * Builds a copy of this {@link CNode} instance where a sub-node
     * designated by flag & a position has been removed.
     * 
     * @param flagPos
     *            a {@link FlagPos} instance
     * @return a copy of this {@link CNode} instance where where a sub-node
     *         designated by flag & a position has been removed.
     */
    public CNode<K, V> removed (final FlagPos flagPos) {
      final BranchNode[] narr = HashTrieMap.removed (this.array, 
                    new BranchNode[this.array.length - 1], 
                    flagPos.position);
      return new CNode<K, V> (narr, this.bitmap ^ flagPos.flag);
    }

    // Build a copy of the current node.
    public CNode<K, V> copied () {
      final BranchNode[] narr = new BranchNode[this.array.length];
      System.arraycopy (this.array, 0, narr, 0, this.array.length);
      return new CNode<K, V> (narr, this.bitmap);
    }

    // Build an empty CNode instance
    CNode () {
      this.array = new BranchNode[] {};
      this.bitmap = 0L;
    }
    
    /**
     * Builds a {@link CNode} instance from a single {@link SNode} instance
     * 
     * @param sNode
     *            a {@link SNode} instance
     * @param width
     *            the width (in power-of-two exponents)
     */
    CNode (final SNode<K, V> sNode, final int width) {
      final long flag = HashTrieMap.flag (sNode.hash (), 0, width);
      this.array = new BranchNode[] { sNode };
      this.bitmap = flag;
    }
    
    /**
     * Builds a {@link CNode} instance from two {@link SNode} objects
     * 
     * @param sn1
     *            a first {@link SNode} instance
     * @param sn2
     *            a second {@link SNode} instance
     * @param level
     *            the current level (in bit progression)
     * @param width
     *            the width (in power-of-two exponents)
     */
    CNode (final SNode<K, V> sn1, final SNode<K, V> sn2, final int level, final int width) {
      final int h1 = sn1.hash ();
      final int h2 = sn2.hash ();
      final long flag1 = HashTrieMap.flag (h1, level, width);
      final long flag2 = HashTrieMap.flag (h2, level, width);
      if (flag1 != flag2) {
	// Make sure the two values are comparable by adding Long.MIN_VALUE so that 
	// indexes 0 & -1 are written in the correct order : 0 and then -1
	if (flag1 + Long.MIN_VALUE < flag2 + Long.MIN_VALUE) {
	  this.array = new BranchNode[] { sn1, sn2 };
	} else {
	  this.array = new BranchNode[] { sn2, sn1 };
	}
      } else {
	// Else goes down one level and create sub nodes
	this.array = new BranchNode[] { new INode (new CNode<K, V> (sn1, sn2, level+width, width)) };
      }
      this.bitmap = flag1 | flag2;
    }

    /**
     * Builds a {@link CNode} from an array of {@link BranchNode} and its
     * computed bitmap.
     * 
     * @param array
     *            the {@link BranchNode} array
     * @param bitmap
     *            the bitmap
     */
    CNode (final BranchNode[] array, final long bitmap) {
      this.array = array;
      this.bitmap = bitmap;
    }
    
    public final BranchNode[] array;    
    public final long bitmap;
  }

  static class KeyValueNode<K, V> {
    KeyValueNode (final K k, final V v) {
      this.key = k;
      this.value = v;
    }
    
    protected final K key;
    protected final V value;
  }

  // A Single Node class, holds a key, a value & a tomb flag.
  static class SingletonSNode<K, V> extends KeyValueNode<K, V> implements SNode<K, V> {
    SingletonSNode (final K k, final V v) {
      super (k, v);
    }
    
    public int hash () {
      return HashTrieMap.hash (this.key);
    }
    
    public TNode<K, V> tombed () {
      return new SingletonTNode<K, V> (this.key, this.value);
    }
    
    public V get (final Object k) {
      if (this.key.equals (k)) {
	return this.value;
      } else {
	return null;
      }
    }
    
    public SNode<K, V> put (final K k, final V v) {
      if (this.key.equals (k)) {
	return new SingletonSNode<K, V> (k, v);
      } else {
	@SuppressWarnings("unchecked")
	  final KeyValueNode<K, V>[] array = new KeyValueNode[] { 
	  new KeyValueNode<K, V> (this.key, this.value), 
	  new KeyValueNode<K, V> (k, v), };
	return new MultiSNode<K, V> (array);
      }
    }
    
    public SNode<K, V> removed (final Object k) {
      return null;
    }
    
    public KeyValueNode<K, V> next (final KeyValueNode<K, V> current) {
      return current == null ? this : null;
    }
  }

  // A Tombed node instance
  static class SingletonTNode<K, V> extends KeyValueNode<K, V> implements TNode<K, V> {
    SingletonTNode (final K k, final V v) {
      super (k, v);
    }
    
    public SNode<K, V> untombed () {
      return new SingletonSNode<K, V> (this.key, this.value);
    }
  }

  // Base class for multiple SNode & TNode implementations
  static class BaseMultiNode<K, V> {
    public BaseMultiNode (final KeyValueNode<K, V>[] array) {
      this.content = array;
    }
    
    protected final KeyValueNode<K, V>[] content;
  }

  // A Multiple key/values SNode
  static class MultiSNode<K, V> extends BaseMultiNode<K, V> implements SNode<K, V> {
    public MultiSNode (final KeyValueNode<K, V>[] content) {
      super (content);
    }
    
    public int hash () {
      return HashTrieMap.hash (this.content [0].key);
    }
    
    public V get (final K k) {
      for (int i = 0; i < this.content.length; i++) {
	final KeyValueNode<K, V> n = this.content [i];
	if (n.key.equals (k)) {
	  return n.value;
	}
      }
      return null;
    }
        
    public SNode<K, V> put (final K k, final V v) {
      int index = -1;
      for (int i = 0; i < this.content.length; i++) {
	final KeyValueNode<K, V> n = this.content [i];
	if (n.key.equals (k)) {
	  index = i;
	  break;
	}
      }
      
      final KeyValueNode<K, V>[] array;
      if (index >= 0) {
	@SuppressWarnings("unchecked")
	  final KeyValueNode<K, V>[] ar = HashTrieMap.updated (
                        this.content, 
                        new KeyValueNode [this.content.length], 
                        new KeyValueNode<K, V> (k, v), 
                        index);
	array = ar;
      } else {
	@SuppressWarnings("unchecked")
	  final KeyValueNode<K, V>[] ar = HashTrieMap.inserted (
                        this.content, 
                        new KeyValueNode [this.content.length + 1], 
                        new KeyValueNode<K, V> (k, v), 
                        this.content.length);
	array = ar;
      }
      return new MultiSNode<K, V> (array);
    }
        
    public SNode<K, V> removed (final Object k) {
      for (int i = 0; i < this.content.length; i++) {
	final KeyValueNode<K, V> n = this.content [i];
	if (n.key.equals (k)) {
	  if (2 == this.content.length) {
	    final KeyValueNode<K, V> kvn = this.content [(i + 1) % 2];
	    return new SingletonSNode<K, V> (kvn.key, kvn.value);
	  } else {
	    @SuppressWarnings("unchecked")
	      final KeyValueNode<K, V>[] narr = HashTrieMap.removed (
                                this.content, 
                                new KeyValueNode [this.content.length - 1], 
                                i);
	    return new MultiSNode<K, V> (narr);
	  }
	}
      }
      throw new RuntimeException ("Key not found:" + k);
    }

    public TNode<K, V> tombed () {
      return new MultiTNode<K, V> (this.content);
    }
        
    public KeyValueNode<K, V> next (final KeyValueNode<K, V> current) {
      if (null == current) {
	return this.content [0];
      } else {
	boolean found = false;
	for (int i = 0; i < this.content.length; i++) {
	  final KeyValueNode<K, V> kvn = this.content [i];
	  if (found) {
	    return kvn;
	  }
	  if (kvn.key.equals (current.key)) {
	    found = true;
	  }
	}
      }
      return null;
    }
  }

  // A Multiple values TNode implementation
  static class MultiTNode<K, V> extends BaseMultiNode<K, V> implements TNode<K, V> {
    public MultiTNode (final KeyValueNode<K, V>[] array) {
      super (array);
    }
    
    public SNode<K, V> untombed () {
      return new MultiSNode<K, V> (this.content);
    }
  }

  // The result of a HashTrieMap.flagPos call is a bit flag & a position
  static class FlagPos {
    FlagPos (final long flag, final int position) {
      this.flag = flag;
      this.position = position;
    }
    
    public final long flag;
    public final int position;
  }

  private static final Constraint<Object> NO_CONSTRAINT = new Constraint<Object> (ConstraintType.NONE, null);
}
