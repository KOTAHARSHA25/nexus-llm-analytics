# Advanced Caching System with Request Deduplication
# Improves performance by caching results and avoiding duplicate computations
# Phase 3.7: Added Semantic Cache for natural language query matching

import time
import hashlib
import json
import logging
import re
from typing import Dict, Any, Optional, Callable, Set, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class CacheEntry:
    """Cache entry with metadata and TTL support"""
    value: Any
    created_at: float
    ttl: float  # Time to live in seconds
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update last accessed time and increment hit count"""
        self.last_accessed = time.time()
        self.hit_count += 1

class AdvancedCache:
    """
    Advanced caching system with features:
    - TTL (Time To Live) support
    - LRU eviction policy
    - Memory limit management
    - Tag-based cache invalidation
    - Request deduplication
    - Performance analytics
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._pending_requests: Dict[str, threading.Event] = {}
        self._request_results: Dict[str, Any] = {}
        
        # Analytics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'duplicate_requests_avoided': 0,
            'total_requests': 0
        }
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate consistent cache key from function parameters"""
        # Create a hashable representation of arguments
        try:
            # Handle complex objects by converting to JSON
            serializable_args = []
            for arg in args:
                if hasattr(arg, '__dict__'):
                    serializable_args.append(str(sorted(arg.__dict__.items())))
                else:
                    serializable_args.append(str(arg))
            
            serializable_kwargs = {}
            for k, v in kwargs.items():
                if hasattr(v, '__dict__'):
                    serializable_kwargs[k] = str(sorted(v.__dict__.items()))
                else:
                    serializable_kwargs[k] = str(v)
            
            cache_data = {
                'function': func_name,
                'args': serializable_args,
                'kwargs': serializable_kwargs
            }
            
            cache_string = json.dumps(cache_data, sort_keys=True)
            return hashlib.sha256(cache_string.encode()).hexdigest()[:16]
            
        except Exception as e:
            logging.warning(f"Cache key generation failed: {e}")
            # Fallback to simple string representation
            return hashlib.sha256(f"{func_name}_{str(args)}_{str(kwargs)}".encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            self._stats['total_requests'] += 1
            
            if key in self._cache:
                entry = self._cache[key]
                
                if entry.is_expired():
                    # Remove expired entry
                    del self._cache[key]
                    self._stats['misses'] += 1
                    return None
                
                # Move to end (LRU)
                self._cache.move_to_end(key)
                entry.touch()
                self._stats['hits'] += 1
                
                logging.debug(f"Cache HIT for key {key}")
                return entry.value
            
            self._stats['misses'] += 1
            logging.debug(f"Cache MISS for key {key}")
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: Set[str] = None):
        """Store value in cache"""
        with self._lock:
            # Use default TTL if not specified
            cache_ttl = ttl if ttl is not None else self.default_ttl
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=cache_ttl,
                tags=tags or set()
            )
            
            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict if necessary
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats['evictions'] += 1
                logging.debug(f"Evicted cache entry {oldest_key}")
            
            logging.debug(f"Cache PUT for key {key} (TTL: {cache_ttl}s)")
    
    def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate all cache entries with matching tags"""
        with self._lock:
            keys_to_remove = []
            for key, entry in self._cache.items():
                if entry.tags.intersection(tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
                logging.debug(f"Invalidated cache entry {key} by tags {tags}")
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._pending_requests.clear()
            self._request_results.clear()
            logging.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self._lock:
            total_requests = self._stats['total_requests']
            hit_rate = (self._stats['hits'] / max(1, total_requests)) * 100
            
            return {
                'hit_rate': round(hit_rate, 2),
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'duplicate_requests_avoided': self._stats['duplicate_requests_avoided'],
                'total_requests': total_requests,
                'current_size': len(self._cache),
                'max_size': self.max_size,
                'memory_efficiency': round((len(self._cache) / max(1, self.max_size)) * 100, 2)
            }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        with self._lock:
            entries_info = []
            for key, entry in list(self._cache.items())[-10:]:  # Last 10 entries
                entries_info.append({
                    'key': key,
                    'created_at': entry.created_at,
                    'ttl': entry.ttl,
                    'hit_count': entry.hit_count,
                    'tags': list(entry.tags),
                    'age_seconds': round(time.time() - entry.created_at, 2),
                    'expires_in': round(entry.ttl - (time.time() - entry.created_at), 2)
                })
            
            return {
                'stats': self.get_stats(),
                'recent_entries': entries_info
            }


class SemanticCache:
    """
    Phase 3.7: Semantic Cache for Natural Language Queries
    
    Uses text similarity to find cached results for semantically similar queries.
    This is useful for LLM-based systems where users may ask the same question
    with different wording.
    
    Features:
    - Jaccard similarity for keyword matching
    - Normalized query comparison
    - Configurable similarity threshold
    - Falls back to exact matching when similarity is low
    """
    
    def __init__(
        self, 
        max_size: int = 200, 
        default_ttl: float = 1800,
        similarity_threshold: float = 0.75
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self._cache: OrderedDict[str, Tuple[str, CacheEntry]] = OrderedDict()  # key -> (original_query, entry)
        self._lock = threading.RLock()
        
        # Common words to ignore in similarity calculation
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'of', 'in', 'to', 'for',
            'on', 'with', 'at', 'by', 'from', 'what', 'which', 'who', 'how',
            'when', 'where', 'why', 'this', 'that', 'these', 'those', 'it', 'its',
            'me', 'my', 'i', 'you', 'your', 'we', 'our', 'they', 'their', 'and',
            'or', 'but', 'if', 'then', 'so', 'just', 'only', 'also', 'please'
        }
        
        # Stats
        self._stats = {
            'hits': 0,
            'semantic_hits': 0,
            'exact_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        logging.info(f"SemanticCache initialized (threshold={similarity_threshold})")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison"""
        # Lowercase
        query = query.lower()
        # Remove punctuation
        query = re.sub(r'[^\w\s]', ' ', query)
        # Normalize whitespace
        query = ' '.join(query.split())
        return query
    
    def _extract_keywords(self, query: str) -> Set[str]:
        """Extract meaningful keywords from query"""
        normalized = self._normalize_query(query)
        words = set(normalized.split())
        # Remove stopwords and short words
        return {w for w in words if w not in self.stopwords and len(w) > 2}
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate Jaccard similarity between two queries.
        Returns a value between 0 (completely different) and 1 (identical).
        """
        keywords1 = self._extract_keywords(query1)
        keywords2 = self._extract_keywords(query2)
        
        if not keywords1 or not keywords2:
            # Fall back to exact match if no keywords
            return 1.0 if self._normalize_query(query1) == self._normalize_query(query2) else 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Bonus for word order similarity (simple n-gram overlap)
        words1 = self._normalize_query(query1).split()
        words2 = self._normalize_query(query2).split()
        
        # Create bigrams
        bigrams1 = set(zip(words1, words1[1:])) if len(words1) > 1 else set()
        bigrams2 = set(zip(words2, words2[1:])) if len(words2) > 1 else set()
        
        bigram_similarity = 0.0
        if bigrams1 and bigrams2:
            bigram_intersection = bigrams1 & bigrams2
            bigram_union = bigrams1 | bigrams2
            bigram_similarity = len(bigram_intersection) / len(bigram_union)
        
        # Weighted combination: 70% keyword overlap, 30% word order
        return 0.7 * jaccard + 0.3 * bigram_similarity
    
    def _generate_key(self, query: str, context: Optional[str] = None) -> str:
        """Generate cache key from query and optional context"""
        normalized = self._normalize_query(query)
        key_data = f"{normalized}_{context or ''}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, query: str, context: Optional[str] = None) -> Optional[Tuple[Any, float]]:
        """
        Get cached result for a query.
        
        Args:
            query: The natural language query
            context: Optional context (e.g., filename) to narrow cache scope
            
        Returns:
            Tuple of (cached_result, similarity_score) or None if no match
        """
        with self._lock:
            self._stats['total_requests'] += 1
            
            # First try exact match
            exact_key = self._generate_key(query, context)
            if exact_key in self._cache:
                original_query, entry = self._cache[exact_key]
                if not entry.is_expired():
                    entry.touch()
                    self._cache.move_to_end(exact_key)
                    self._stats['hits'] += 1
                    self._stats['exact_hits'] += 1
                    logging.debug(f"SemanticCache EXACT HIT for: {query[:50]}...")
                    return (entry.value, 1.0)
                else:
                    del self._cache[exact_key]
            
            # Try semantic match
            best_match = None
            best_similarity = 0.0
            best_key = None
            
            for key, (stored_query, entry) in list(self._cache.items()):
                if entry.is_expired():
                    continue
                
                # Only compare queries with same context
                stored_context = entry.tags.get('context') if hasattr(entry.tags, 'get') else None
                if context != stored_context:
                    continue
                
                similarity = self._calculate_similarity(query, stored_query)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = entry
                    best_key = key
            
            if best_match and best_key:
                best_match.touch()
                self._cache.move_to_end(best_key)
                self._stats['hits'] += 1
                self._stats['semantic_hits'] += 1
                logging.info(f"SemanticCache SEMANTIC HIT (similarity={best_similarity:.2f}) for: {query[:50]}...")
                return (best_match.value, best_similarity)
            
            self._stats['misses'] += 1
            logging.debug(f"SemanticCache MISS for: {query[:50]}...")
            return None
    
    def put(
        self, 
        query: str, 
        value: Any, 
        context: Optional[str] = None,
        ttl: Optional[float] = None
    ):
        """Store a query result in the cache"""
        with self._lock:
            key = self._generate_key(query, context)
            cache_ttl = ttl if ttl is not None else self.default_ttl
            
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=cache_ttl,
                tags={'context': context} if context else set()
            )
            
            self._cache[key] = (query, entry)
            self._cache.move_to_end(key)
            
            # Evict if necessary (LRU)
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            logging.debug(f"SemanticCache PUT for: {query[:50]}... (TTL: {cache_ttl}s)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self._stats['total_requests']
            hits = self._stats['hits']
            
            return {
                'hit_rate': round((hits / max(1, total)) * 100, 2),
                'total_requests': total,
                'hits': hits,
                'exact_hits': self._stats['exact_hits'],
                'semantic_hits': self._stats['semantic_hits'],
                'misses': self._stats['misses'],
                'current_size': len(self._cache),
                'max_size': self.max_size,
                'similarity_threshold': self.similarity_threshold
            }
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            self._cache.clear()
            logging.info("SemanticCache cleared")

# Global cache instances
_query_cache = AdvancedCache(max_size=500, default_ttl=1800)  # 30 minutes
_model_cache = AdvancedCache(max_size=100, default_ttl=3600)  # 1 hour
_file_analysis_cache = AdvancedCache(max_size=200, default_ttl=7200)  # 2 hours
_semantic_cache = SemanticCache(max_size=200, default_ttl=1800, similarity_threshold=0.75)  # Phase 3.7

def cached_query(ttl: float = 1800, tags: Set[str] = None):
    """Decorator for caching query results with deduplication"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _query_cache._generate_key(func.__name__, args, kwargs)
            
            # Check if request is already in progress (deduplication)
            with _query_cache._lock:
                if cache_key in _query_cache._pending_requests:
                    # Wait for the ongoing request
                    _query_cache._stats['duplicate_requests_avoided'] += 1
                    logging.info(f"Deduplicating request for {func.__name__}")
                    
                    # Wait for result
                    event = _query_cache._pending_requests[cache_key]
                    event.wait(timeout=300)  # 5 minute timeout
                    
                    # Get result
                    if cache_key in _query_cache._request_results:
                        result = _query_cache._request_results[cache_key]
                        del _query_cache._request_results[cache_key]
                        return result
                
                # Check cache first
                cached_result = _query_cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Mark request as pending
                _query_cache._pending_requests[cache_key] = threading.Event()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache successful results
                if isinstance(result, dict) and not result.get('error'):
                    _query_cache.put(cache_key, result, ttl=ttl, tags=tags)
                
                # Store result for other waiting threads
                with _query_cache._lock:
                    _query_cache._request_results[cache_key] = result
                    if cache_key in _query_cache._pending_requests:
                        _query_cache._pending_requests[cache_key].set()
                
                return result
                
            except Exception as e:
                # Signal error to waiting threads
                with _query_cache._lock:
                    error_result = {"error": str(e), "status": "error"}
                    _query_cache._request_results[cache_key] = error_result
                    if cache_key in _query_cache._pending_requests:
                        _query_cache._pending_requests[cache_key].set()
                raise
            
            finally:
                # Cleanup pending request
                with _query_cache._lock:
                    if cache_key in _query_cache._pending_requests:
                        del _query_cache._pending_requests[cache_key]
        
        return wrapper
    return decorator

def cached_model_operation(ttl: float = 3600):
    """Decorator for caching model operations"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = _model_cache._generate_key(func.__name__, args, kwargs)
            
            # Check cache
            cached_result = _model_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute and cache
            result = func(*args, **kwargs)
            if isinstance(result, dict) and not result.get('error'):
                _model_cache.put(cache_key, result, ttl=ttl, tags={'models'})
            
            return result
        return wrapper
    return decorator

def cached_file_analysis(ttl: float = 7200):
    """Decorator for caching file analysis results"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = _file_analysis_cache._generate_key(func.__name__, args, kwargs)
            
            # Check cache
            cached_result = _file_analysis_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute and cache
            result = func(*args, **kwargs)
            if isinstance(result, dict) and not result.get('error'):
                _file_analysis_cache.put(cache_key, result, ttl=ttl, tags={'file_analysis'})
            
            return result
        return wrapper
    return decorator


def semantic_cached_query(ttl: float = 1800, context_extractor: Callable = None):
    """
    Phase 3.7: Decorator for caching natural language query results with semantic matching.
    
    Args:
        ttl: Time-to-live in seconds
        context_extractor: Optional function to extract context from args/kwargs
            (e.g., lambda args, kwargs: kwargs.get('filename'))
    
    Example:
        @semantic_cached_query(ttl=1800, context_extractor=lambda a, k: k.get('file'))
        def analyze_query(query: str, file: str = None):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract the query (assumed to be first arg or 'query' kwarg)
            query = None
            if args:
                query = str(args[0])
            elif 'query' in kwargs:
                query = str(kwargs['query'])
            
            if not query:
                # Can't use semantic cache without a query, run function directly
                return func(*args, **kwargs)
            
            # Extract context if extractor provided
            context = None
            if context_extractor:
                try:
                    context = context_extractor(args, kwargs)
                except (IndexError, KeyError, TypeError):
                    pass
            
            # Check semantic cache
            cached = _semantic_cache.get(query, context=context)
            if cached is not None:
                result, similarity = cached
                logging.debug(f"Semantic cache hit for {func.__name__} (similarity: {similarity:.2f})")
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache successful results
            if isinstance(result, dict) and not result.get('error'):
                _semantic_cache.put(query, result, context=context, ttl=ttl)
            
            return result
        return wrapper
    return decorator

def invalidate_model_cache():
    """Invalidate model-related cache entries"""
    _model_cache.invalidate_by_tags({'models'})
    logging.info("Model cache invalidated")

def invalidate_file_cache():
    """Invalidate file analysis cache entries"""
    _file_analysis_cache.invalidate_by_tags({'file_analysis'})
    logging.info("File analysis cache invalidated")

def get_cache_status() -> Dict[str, Any]:
    """Get status of all cache systems"""
    return {
        'query_cache': _query_cache.get_stats(),
        'model_cache': _model_cache.get_stats(),
        'file_analysis_cache': _file_analysis_cache.get_stats(),
        'semantic_cache': _semantic_cache.get_stats(),
        'overall_performance': {
            'total_requests': (_query_cache._stats['total_requests'] + 
                             _model_cache._stats['total_requests'] + 
                             _file_analysis_cache._stats['total_requests'] +
                             _semantic_cache._stats['total_requests']),
            'total_hits': (_query_cache._stats['hits'] + 
                          _model_cache._stats['hits'] + 
                          _file_analysis_cache._stats['hits'] +
                          _semantic_cache._stats['hits']),
            'duplicate_requests_avoided': (_query_cache._stats['duplicate_requests_avoided'] + 
                                         _model_cache._stats['duplicate_requests_avoided'] + 
                                         _file_analysis_cache._stats['duplicate_requests_avoided']),
            'semantic_cache_hits': _semantic_cache._stats['semantic_hits']
        }
    }

def clear_all_caches():
    """Clear all cache systems"""
    _query_cache.clear()
    _model_cache.clear()
    _file_analysis_cache.clear()
    _semantic_cache.clear()
    logging.info("All caches cleared")


def get_semantic_cache() -> SemanticCache:
    """Get the global semantic cache instance for direct access"""
    return _semantic_cache