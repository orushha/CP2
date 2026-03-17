import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiConsumer;

// ----------------------------------------------------------------------
// A wrapper around the Java class library's sophisticated
// ConcurrentHashMap<K,V>, making it implement OurMap<K,V>

public class WrapConcurrentHashMap<K,V> implements OurMap<K,V> {
  final ConcurrentHashMap<K,V> underlying = new ConcurrentHashMap<K,V>();

  public boolean containsKey(K k) {
    return underlying.containsKey(k);
  }

  public V get(K k) {
    return underlying.get(k);
  }

  public V put(K k, V v) {
    return underlying.put(k, v);
  }

  public V remove(K k) {
    return underlying.remove(k);
  }

  public int size() {
    return underlying.size();
  }
  
  public void forEach(BiConsumer<K,V> consumer) {
    underlying.forEach((k,v) -> consumer.accept(k,v));
  }
}
