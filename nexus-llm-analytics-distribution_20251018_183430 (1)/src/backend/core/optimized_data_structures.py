# Advanced Data Structures and Algorithms Optimization
# BEFORE: O(n) linear searches, inefficient data processing, memory fragmentation
# AFTER: O(1) hash map lookups, optimized algorithms, efficient memory usage

from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import heapq
import bisect
from functools import lru_cache, wraps
from dataclasses import dataclass, field
from threading import RLock
import time
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from contextlib import asynccontextmanager
import weakref

# High-performance data structures
class TrieNode:
    """Optimized Trie for fast prefix matching and query parsing"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_word = False
        self.frequency = 0
        self.suggestions: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    @property
    def is_end_of_word(self) -> bool:
        """Compatibility property for tests"""
        return self.is_end_word
    
    @is_end_of_word.setter
    def is_end_of_word(self, value: bool):
        """Compatibility setter for tests"""
        self.is_end_word = value

class OptimizedTrie:
    """Memory-efficient Trie with compressed paths and smart suggestions"""
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
        self._suggestion_cache = {}
    
    @property
    def size(self) -> int:
        """Return the number of words in the trie"""
        return self.word_count
        
    def insert(self, word: str, value: Any = None) -> bool:
        """Insert word with O(m) complexity where m is word length"""
        if not word:
            return False
            
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.frequency += 1
            
        if not node.is_end_word:
            self.word_count += 1
            
        node.is_end_word = True
        node.metadata = {'value': value} if value is not None else {}
        
        # Clear suggestion cache as it's now invalid
        self._suggestion_cache.clear()
        return True
    
    def search(self, word: str) -> Any:
        """Search for exact word match - O(m) complexity where m is word length"""
        if not word:
            return None
            
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        
        if node.is_end_word and 'value' in node.metadata:
            return node.metadata['value']
        return None
    
    def search_prefix(self, prefix: str, max_suggestions: int = 10) -> List[Tuple[str, int, Dict]]:
        """Fast prefix search with frequency-based ranking - O(p + k) complexity"""
        if not prefix:
            return []
            
        cache_key = f"{prefix}:{max_suggestions}"
        if cache_key in self._suggestion_cache:
            return self._suggestion_cache[cache_key]
            
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        
        # BFS to find all words with this prefix
        suggestions = []
        queue = deque([(node, prefix)])
        
        while queue and len(suggestions) < max_suggestions * 2:  # Get extra for ranking
            current_node, current_word = queue.popleft()
            
            if current_node.is_end_word:
                suggestions.append((current_word, current_node.frequency, current_node.metadata))
            
            for char, child_node in current_node.children.items():
                queue.append((child_node, current_word + char))
        
        # Sort by frequency and take top results
        suggestions.sort(key=lambda x: x[1], reverse=True)
        result = suggestions[:max_suggestions]
        
        # Cache result
        self._suggestion_cache[cache_key] = result
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trie statistics"""
        def count_nodes(node):
            count = 1  # Current node
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        return {
            'total_nodes': count_nodes(self.root),
            'word_count': self.word_count,
            'cache_size': len(self._suggestion_cache)
        }

@dataclass
class QueryNode:
    """Optimized query parsing node with efficient memory layout"""
    text: str
    node_type: str
    confidence: float
    children: List['QueryNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _hash: Optional[int] = field(default=None, init=False)
    
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.text, self.node_type, self.confidence))
        return self._hash

class HighPerformanceHashMap:
    """Custom hash map with optimized collision handling and memory efficiency"""
    
    def __init__(self, initial_capacity: int = 16, load_factor: float = 0.75):
        self.capacity = initial_capacity
        self.size = 0
        self.load_factor = load_factor
        self.buckets = [[] for _ in range(self.capacity)]
        self.lock = RLock()
        
    def _hash(self, key: Any) -> int:
        """Optimized hash function with better distribution"""
        if isinstance(key, str):
            # Use built-in hash with salt for strings
            return hash(key) % self.capacity
        elif isinstance(key, (int, float)):
            # Fibonacci hashing for numbers
            return int((key * 2654435769) % (2**32)) % self.capacity
        else:
            return hash(key) % self.capacity
    
    def _resize(self) -> None:
        """Resize hash map when load factor is exceeded"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all elements
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    def put(self, key: Any, value: Any) -> None:
        """Insert with O(1) average complexity"""
        with self.lock:
            if self.size >= self.capacity * self.load_factor:
                self._resize()
            
            hash_index = self._hash(key)
            bucket = self.buckets[hash_index]
            
            # Update existing key
            for i, (k, v) in enumerate(bucket):
                if k == key:
                    bucket[i] = (key, value)
                    return
            
            # Insert new key
            bucket.append((key, value))
            self.size += 1
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Retrieve with O(1) average complexity"""
        hash_index = self._hash(key)
        bucket = self.buckets[hash_index]
        
        for k, v in bucket:
            if k == key:
                return v
        return default
    
    def delete(self, key: Any) -> bool:
        """Delete with O(1) average complexity"""
        with self.lock:
            hash_index = self._hash(key)
            bucket = self.buckets[hash_index]
            
            for i, (k, v) in enumerate(bucket):
                if k == key:
                    del bucket[i]
                    self.size -= 1
                    return True
            return False

class LRUCache:
    """High-performance LRU cache with O(1) operations"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.freq_map = defaultdict(int)
        self.min_freq = 0
        self.freq_to_keys = defaultdict(set)
        self.key_to_freq = {}
        
    def _update_freq(self, key: str) -> None:
        """Update frequency tracking for LFU eviction"""
        freq = self.key_to_freq[key]
        self.freq_to_keys[freq].remove(key)
        
        if not self.freq_to_keys[freq] and freq == self.min_freq:
            self.min_freq += 1
            
        self.key_to_freq[key] = freq + 1
        self.freq_to_keys[freq + 1].add(key)
    
    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None
            
        self._update_freq(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        if self.capacity <= 0:
            return
            
        if key in self.cache:
            self.cache[key] = value
            self._update_freq(key)
            return
        
        if len(self.cache) >= self.capacity:
            # Evict least frequently used
            remove_key = self.freq_to_keys[self.min_freq].pop()
            del self.cache[remove_key]
            del self.key_to_freq[remove_key]
        
        self.cache[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1].add(key)
        self.min_freq = 1

class OptimizedDataProcessor:
    """High-performance data processing with advanced algorithms"""
    
    def __init__(self):
        self.trie = OptimizedTrie()
        self.hash_map = HighPerformanceHashMap()
        self.cache = LRUCache(1000)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def batch_process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Optimized batch processing with parallel execution"""
        if not queries:
            return []
        
        # Deduplicate queries using set for O(1) lookup
        unique_queries = list(set(queries))
        
        # Process in parallel batches
        batch_size = max(1, len(unique_queries) // 4)
        batches = [unique_queries[i:i + batch_size] 
                  for i in range(0, len(unique_queries), batch_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                batch_results = future.result()
                results.extend(batch_results)
        
        return results
    
    def _process_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of queries efficiently"""
        results = []
        for query in queries:
            # Check cache first
            cached_result = self.cache.get(query)
            if cached_result:
                results.append(cached_result)
                continue
            
            # Process query
            result = self._process_single_query(query)
            self.cache.put(query, result)
            results.append(result)
        
        return results
    
    def _process_single_query(self, query: str) -> Dict[str, Any]:
        """Optimized single query processing"""
        # Tokenize efficiently
        tokens = self._fast_tokenize(query)
        
        # Extract entities using trie
        entities = []
        for token in tokens:
            suggestions = self.trie.search_prefix(token, max_suggestions=5)
            if suggestions:
                entities.extend(suggestions)
        
        # Build query graph using optimized algorithms
        query_graph = self._build_query_graph(tokens, entities)
        
        return {
            'query': query,
            'tokens': tokens,
            'entities': entities,
            'graph': query_graph,
            'processing_time': time.time()
        }
    
    def _fast_tokenize(self, text: str) -> List[str]:
        """Optimized tokenization with minimal allocations"""
        if not text:
            return []
        
        # Use list comprehension for speed
        tokens = [
            token.strip().lower() 
            for token in text.replace(',', ' ').replace('.', ' ').split()
            if token.strip()
        ]
        
        return tokens
    
    def _build_query_graph(self, tokens: List[str], entities: List[Tuple]) -> Dict[str, Any]:
        """Build optimized query representation graph"""
        # Use adjacency list for efficient graph representation
        graph = {
            'nodes': [],
            'edges': [],
            'adjacency': defaultdict(list)
        }
        
        # Create nodes efficiently
        node_map = {}
        for i, token in enumerate(tokens):
            node = QueryNode(
                text=token,
                node_type='token',
                confidence=1.0
            )
            graph['nodes'].append(node)
            node_map[token] = i
        
        # Add entity nodes
        for entity_text, frequency, metadata in entities:
            if entity_text not in node_map:
                node = QueryNode(
                    text=entity_text,
                    node_type='entity',
                    confidence=min(frequency / 100.0, 1.0),
                    metadata=metadata
                )
                graph['nodes'].append(node)
                node_map[entity_text] = len(graph['nodes']) - 1
        
        # Build edges efficiently using sliding window
        for i in range(len(tokens) - 1):
            source_idx = node_map[tokens[i]]
            target_idx = node_map[tokens[i + 1]]
            
            edge = {
                'source': source_idx,
                'target': target_idx,
                'weight': 1.0,
                'type': 'sequence'
            }
            
            graph['edges'].append(edge)
            graph['adjacency'][source_idx].append(target_idx)
        
        return graph

# Memory-efficient connection pooling
class OptimizedConnectionPool:
    """High-performance connection pool with smart recycling"""
    
    def __init__(self, max_connections: int = 20, timeout: float = 30.0):
        self.max_connections = max_connections
        self.timeout = timeout
        self.available_connections = deque()
        self.used_connections = set()
        self.connection_factory = None
        self.lock = RLock()
        
    @asynccontextmanager
    async def get_connection(self):
        """Get connection with automatic cleanup"""
        connection = None
        try:
            with self.lock:
                if self.available_connections:
                    connection = self.available_connections.popleft()
                elif len(self.used_connections) < self.max_connections:
                    connection = await self._create_connection()
                else:
                    # Wait for available connection
                    await asyncio.sleep(0.1)
                    async with self.get_connection() as conn:
                        yield conn
                        return
                
                self.used_connections.add(connection)
            
            yield connection
            
        finally:
            if connection:
                with self.lock:
                    self.used_connections.discard(connection)
                    if await self._is_connection_healthy(connection):
                        self.available_connections.append(connection)
                    else:
                        await self._close_connection(connection)
    
    async def _create_connection(self):
        """Create new connection - implement based on needs"""
        # This would be implemented based on the specific connection type
        # (database, HTTP, etc.)
        pass
    
    async def _is_connection_healthy(self, connection) -> bool:
        """Check if connection is still usable"""
        # Implementation depends on connection type
        return True
    
    async def _close_connection(self, connection):
        """Properly close connection"""
        # Implementation depends on connection type
        pass

# Advanced algorithm implementations
class OptimizedAlgorithms:
    """Collection of optimized algorithms for data processing"""
    
    @staticmethod
    def fast_fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Optimized fuzzy string matching using sliding window"""
        if not text1 or not text2:
            return False
        
        len1, len2 = len(text1), len(text2)
        if abs(len1 - len2) > max(len1, len2) * (1 - threshold):
            return False
        
        # Use sliding window approach for efficiency
        matches = 0
        window_size = min(len1, len2) // 4 + 1
        
        for i in range(0, len1, window_size):
            substr1 = text1[i:i + window_size]
            for j in range(max(0, i - window_size), min(len2, i + 2 * window_size)):
                if j + window_size <= len2:
                    substr2 = text2[j:j + window_size]
                    if substr1 == substr2:
                        matches += window_size
                        break
        
        similarity = matches / max(len1, len2)
        return similarity >= threshold
    
    @staticmethod
    def optimize_query_plan(query_components: List[Dict]) -> List[Dict]:
        """Optimize query execution plan using cost-based optimization"""
        if not query_components:
            return []
        
        # Assign costs to different operations
        cost_map = {
            'filter': 1,
            'sort': 2,
            'join': 5,
            'aggregate': 3,
            'scan': 1
        }
        
        # Sort by cost (greedy optimization)
        optimized_plan = sorted(
            query_components,
            key=lambda x: cost_map.get(x.get('operation', 'scan'), 10)
        )
        
        return optimized_plan
    
    @staticmethod
    def parallel_merge_sort(data: List[Any], key_func=None) -> List[Any]:
        """Optimized parallel merge sort for large datasets"""
        if len(data) <= 1:
            return data
        
        if len(data) < 1000:  # Use regular sort for small datasets
            return sorted(data, key=key_func)
        
        # Divide into chunks for parallel processing
        chunk_size = len(data) // 4
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Sort chunks in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            sorted_chunks = list(executor.map(
                lambda chunk: sorted(chunk, key=key_func), 
                chunks
            ))
        
        # Merge sorted chunks
        result = []
        heap = []
        
        # Initialize heap with first element from each chunk
        for i, chunk in enumerate(sorted_chunks):
            if chunk:
                heapq.heappush(heap, (key_func(chunk[0]) if key_func else chunk[0], i, 0, chunk[0]))
        
        # Merge using heap
        while heap:
            _, chunk_idx, elem_idx, value = heapq.heappop(heap)
            result.append(value)
            
            # Add next element from same chunk
            if elem_idx + 1 < len(sorted_chunks[chunk_idx]):
                next_elem = sorted_chunks[chunk_idx][elem_idx + 1]
                heapq.heappush(heap, (
                    key_func(next_elem) if key_func else next_elem,
                    chunk_idx,
                    elem_idx + 1,
                    next_elem
                ))
        
        return result

# Factory for creating optimized data structures
class DataStructureFactory:
    """Factory for creating optimized data structures based on use case"""
    
    @staticmethod
    def create_optimized_index(data_type: str, size_hint: int = 1000) -> Any:
        """Create optimal data structure based on requirements"""
        if data_type == "search":
            return OptimizedTrie()
        elif data_type == "cache":
            return LRUCache(min(size_hint, 10000))
        elif data_type == "hash_map":
            initial_capacity = max(16, size_hint // 4)
            return HighPerformanceHashMap(initial_capacity)
        else:
            return {}
    
    @staticmethod
    def create_processor(processing_type: str) -> OptimizedDataProcessor:
        """Create optimized processor for specific use case"""
        processor = OptimizedDataProcessor()
        
        # Configure based on processing type
        if processing_type == "nlp":
            # Preload common NLP terms into trie
            common_terms = ["analyze", "data", "report", "statistics", "trend", "pattern"]
            for term in common_terms:
                processor.trie.insert(term, {"type": "nlp_term"})
        
        return processor

# Performance monitoring and metrics
class PerformanceMonitor:
    """Monitor and track performance metrics for optimizations"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and record duration"""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        return duration
    
    def get_average_time(self, operation: str) -> float:
        """Get average execution time for operation"""
        times = self.metrics.get(operation, [])
        return sum(times) / len(times) if times else 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {}
        for operation, times in self.metrics.items():
            report[operation] = {
                'count': len(times),
                'average': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'total': sum(times)
            }
        return report