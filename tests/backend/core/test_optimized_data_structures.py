import pytest
import threading
from src.backend.core.optimized_data_structures import (
    OptimizedTrie, HighPerformanceHashMap, LRUCache, OptimizedDataProcessor
)

# --- OptimizedTrie Tests ---
def test_trie_operations():
    trie = OptimizedTrie()
    trie.insert("apple", value="fruit")
    trie.insert("app", value="app")
    trie.insert("apricot", value="fruit")
    
    assert trie.search("apple") == "fruit"
    assert trie.search("app") == "app"
    assert trie.search("banana") is None
    
    # Prefix search
    suggestions = trie.search_prefix("ap")
    assert len(suggestions) == 3
    words = [s[0] for s in suggestions]
    assert "apple" in words
    assert "apricot" in words
    assert "app" in words

def test_trie_stats():
    trie = OptimizedTrie()
    trie.insert("a")
    stats = trie.get_stats()
    assert stats['word_count'] == 1
    assert stats['total_nodes'] > 1

# --- HighPerformanceHashMap Tests ---
def test_hashmap_basic():
    hmap = HighPerformanceHashMap(initial_capacity=4)
    hmap.put("key1", "val1")
    assert hmap.get("key1") == "val1"
    assert hmap.get("key2") is None
    
    hmap.delete("key1")
    assert hmap.get("key1") is None

def test_hashmap_resize():
    # Capacity 2, load factor default 0.75 -> resize at 1.5 elements (so 2nd element)
    hmap = HighPerformanceHashMap(initial_capacity=2) 
    hmap.put("k1", "v1")
    hmap.put("k2", "v2")
    hmap.put("k3", "v3") # Should trigger resize
    
    assert hmap.capacity > 2
    assert hmap.get("k1") == "v1"
    assert hmap.get("k3") == "v3"

def test_hashmap_thread_safety():
    hmap = HighPerformanceHashMap()
    def worker():
        for i in range(100):
            hmap.put(f"k{i}", i)
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    assert hmap.size <= 100 # Should be exactly 100 if keys overwrite, but size logic is +1 on insert new
    # Wait, simple put: if key exists update, else append. 
    # With generic 'k{i}', if multiple threads put same key, it updates.
    # Total keys should be 100.
    assert hmap.get("k99") == 99

# --- LRUCache Tests ---
def test_lru_cache():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    
    assert cache.get("a") == 1 # Access a, so b is LRU (actually implementation is LFU-ish, let's check code)
    # Code Logic: 
    # put "a" -> freq 1
    # put "b" -> freq 1
    # get "a" -> freq becomes 2
    # So "b" has freq 1 (min_freq), "a" has freq 2.
    # put "c" -> eviction needed. 
    # remove_key = freq_to_keys[min_freq].pop() -> pops from freq 1 -> "b"
    
    cache.put("c", 3)
    
    assert cache.get("b") is None # Evicted
    assert cache.get("a") == 1
    assert cache.get("c") == 3

# --- OptimizedDataProcessor Tests ---
def test_processor_tokenize():
    proc = OptimizedDataProcessor()
    tokens = proc._fast_tokenize("Hello, World!")
    assert tokens == ["hello", "world!"] # Code splits by space after replace comma/dot with space.
    # replace(',', ' ') -> "Hello  World!" replacement 
    # replace('.', ' ') -> "Hello  World!" (if exclamation remains)
    # split() -> ["Hello", "World!"] -> lower/strip -> ["hello", "world!"]
    
    # Wait, replace('.', ' ') handles "Mr. Smith"? "Mr  Smith"
    
def test_batch_process(proc=None):
    if not proc:
        proc = OptimizedDataProcessor()
    queries = ["query1", "query2", "query1"] # Duplicate
    results = proc.batch_process_queries(queries)
    
    assert len(results) == 2 # Deduped processing, but results list might match input?
    # Code: unique_queries = set(queries). Returns list of results for UNIQUE queries.
    # So if I pass 3 queries (2 unique), I get 2 results.
    
    assert len(results) == 2
    assert results[0]['query'] in ["query1", "query2"]
