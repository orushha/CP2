// Interface for hash map implementations
// sestoft@itu.dk * 2014

import java.util.function.BiConsumer;

public interface OurMap<K,V> {
  boolean containsKey(K k);
  V get(K k);
  V put(K k, V v);
  V remove(K k);
  int size();
  void forEach(BiConsumer<K,V> consumer);
}
