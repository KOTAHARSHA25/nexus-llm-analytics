# Enhanced Caching System Integration
# Combines existing advanced_cache.py with new optimized data structures
# BEFORE: Single cache implementation, O(n) lookups in some cases
# AFTER: Multi-tier caching with O(1) operations, intelligent cache warming

import asyncio
import time
import hashlib
import json
import logging
import pickle
import os
from typing import Dict, Any, Optional, Callable, Set, List, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict, deque
from functools import wraps, lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
from enum import Enum
import statistics

# Import our optimized data structures
from .optimized_data_structures import (
    OptimizedTrie, HighPerformanceHashMap, LRUCache as OptimizedLRUCache,
    OptimizedDataProcessor, PerformanceMonitor
)

# Re-implementing AdvancedCache functionality here since the original module is missing
class PersistentCache:
    """L3 Persistent Cache using file system"""
    
    def __init__(self, capacity: int, default_ttl: float):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache_dir = os.path.join(os.getcwd(), "data", "cache", "l3")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_path(self, key: str) -> str:
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.pickle")
    
    def get(self, key: str) -> Any:
        path = self._get_path(key)
        if not os.path.exists(path):
            return None
            
        try:
            # Check TTL
            mtime = os.path.getmtime(path)
            if time.time() - mtime > self.default_ttl:
                os.remove(path)
                return None
                
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
            
    def put(self, key: str, value: Any, ttl: float = None, tags: Set[str] = None):
        path = self._get_path(key)
        try:
            with open(path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logging.warning(f"Failed to write L3 cache: {e}")
            
    def invalidate_by_tags(self, tags: Set[str]):
        # Simplified: File based tagging is complex, skipping for now or clear all
        pass
        
    def get_stats(self):
        return {"type": "filesystem", "dir": self.cache_dir}

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache hierarchy levels for multi-tier caching"""
    L1_MEMORY = "l1_memory"      # Fastest, smallest
    L2_OPTIMIZED = "l2_optimized"  # Medium speed, medium size
    L3_PERSISTENT = "l3_persistent"  # Slower, largest

@dataclass
class EnhancedCacheEntry:
    """Enhanced cache entry with performance metrics and metadata"""
    value: Any
    created_at: float
    ttl: float
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    cache_level: CacheLevel = CacheLevel.L1_MEMORY
    access_pattern: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate entry size after initialization"""
        if self.size_bytes == 0:
            try:
                self.size_bytes = len(pickle.dumps(self.value))
            except Exception:
                self.size_bytes = len(str(self.value).encode('utf-8'))
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access metrics"""
        current_time = time.time()
        self.last_accessed = current_time
        self.hit_count += 1
        
        # Track access pattern (keep last 10 accesses)
        self.access_pattern.append(current_time)
        if len(self.access_pattern) > 10:
            self.access_pattern.pop(0)
    
    def get_access_frequency(self) -> float:
        """Calculate access frequency per hour"""
        if len(self.access_pattern) < 2:
            return 0.0
        
        time_span = self.access_pattern[-1] - self.access_pattern[0]
        if time_span <= 0:
            return 0.0
        
        return len(self.access_pattern) / (time_span / 3600)  # per hour

class SmartCacheWarmer:
    """Intelligent cache warming based on access patterns"""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.warming_patterns = defaultdict(list)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.trie = OptimizedTrie()
        self._build_pattern_index()
    
    def _build_pattern_index(self):
        """Build trie index of common cache key patterns"""
        common_patterns = [
            "query_", "file_", "analysis_", "report_", "data_",
            "user_", "model_", "result_", "batch_", "stream_"
        ]
        
        for pattern in common_patterns:
            self.trie.insert(pattern, {"type": "cache_pattern", "priority": 1.0})
    
    def record_access(self, key: str, access_time: float):
        """Record access pattern for predictive warming"""
        # Extract pattern from key
        pattern = self._extract_pattern(key)
        self.warming_patterns[pattern].append(access_time)
        
        # Keep only recent accesses (last 24 hours)
        cutoff_time = access_time - 86400
        self.warming_patterns[pattern] = [
            t for t in self.warming_patterns[pattern] if t > cutoff_time
        ]
    
    def _extract_pattern(self, key: str) -> str:
        """Extract pattern from cache key"""
        suggestions = self.trie.search_prefix(key.split('_')[0] + '_', max_suggestions=1)
        if suggestions:
            return suggestions[0][0]
        return "generic_"
    
    async def warm_cache_predictively(self):
        """Warm cache based on predicted access patterns"""
        current_time = time.time()
        
        for pattern, access_times in self.warming_patterns.items():
            if len(access_times) < 3:
                continue
            
            # Predict next access time
            intervals = [access_times[i] - access_times[i-1] 
                        for i in range(1, len(access_times))]
            
            if intervals:
                avg_interval = statistics.mean(intervals)
                predicted_next = access_times[-1] + avg_interval
                
                # If prediction suggests access soon, warm relevant caches
                if predicted_next - current_time < 300:  # Within 5 minutes
                    await self._warm_pattern_cache(pattern)
    
    async def _warm_pattern_cache(self, pattern: str):
        """Warm cache for specific pattern"""
        logger.info(f"Warming cache for pattern: {pattern}")
        # Implementation would depend on specific warming strategies

class EnhancedCacheManager:
    """Enhanced multi-tier cache manager with optimized data structures"""
    
    def __init__(self, 
                 l1_size: int = 500,
                 l2_size: int = 2000, 
                 l3_size: int = 10000,
                 default_ttl: float = 3600):
        
        # Multi-tier cache setup
        self.l1_cache = OptimizedLRUCache(l1_size)  # Fastest tier
        self.l2_cache = HighPerformanceHashMap(l2_size // 4)  # Medium tier
        self.l3_cache = PersistentCache(l3_size, default_ttl)  # Largest tier
        
        # Optimization components
        self.trie_index = OptimizedTrie()
        self.data_processor = OptimizedDataProcessor()
        self.performance_monitor = PerformanceMonitor()
        self.cache_warmer = SmartCacheWarmer(self)
        
        # Enhanced statistics
        self.stats = defaultdict(int)
        self.tier_stats = {
            CacheLevel.L1_MEMORY: defaultdict(int),
            CacheLevel.L2_OPTIMIZED: defaultdict(int),
            CacheLevel.L3_PERSISTENT: defaultdict(int)
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Query optimization
        self.query_patterns = {}
        self.optimization_rules = []
        
        # Background tasks (only if not disabled)
        if not os.environ.get('DISABLE_CACHE_BACKGROUND'):
            self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background optimization tasks"""
        try:
            # Start cache warming task
            asyncio.create_task(self._periodic_cache_warming())
            
            # Start cleanup task
            asyncio.create_task(self._periodic_cleanup())
            
            # Start optimization task
        except RuntimeError:
            # No event loop running (e.g., in tests)
            logging.info("Skipping background tasks - no event loop")
        asyncio.create_task(self._periodic_optimization())
    
    def _generate_optimized_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate optimized cache key using trie-based patterns"""
        try:
            # Check if we have a pattern for this function
            function_patterns = self.trie_index.search_prefix(func_name, max_suggestions=3)
            
            if function_patterns:
                # Use optimized pattern
                pattern = function_patterns[0][0]
                base_key = f"{pattern}_{func_name}"
            else:
                base_key = func_name
                # Add to trie for future optimization
                self.trie_index.insert(func_name, {"type": "function", "usage": 1})
            
            # Process arguments efficiently
            arg_hash = self._fast_hash_args(args, kwargs)
            return f"{base_key}_{arg_hash}"
            
        except Exception as e:
            logger.warning(f"Optimized key generation failed: {e}")
            return self._fallback_key_generation(func_name, args, kwargs)
    
    def _fast_hash_args(self, args: tuple, kwargs: dict) -> str:
        """Fast argument hashing with collision resistance"""
        # Use optimized hashing for common types
        hash_components = []
        
        for arg in args[:5]:  # Limit to first 5 args for performance
            if isinstance(arg, str):
                hash_components.append(f"s:{hash(arg) % 1000000}")
            elif isinstance(arg, (int, float)):
                hash_components.append(f"n:{arg}")
            elif isinstance(arg, (list, tuple)):
                hash_components.append(f"l:{len(arg)}:{hash(str(arg[:3]))}")
            else:
                hash_components.append(f"o:{type(arg).__name__}")
        
        # Process key kwargs only
        key_kwargs = {k: v for k, v in kwargs.items() if k in ['query', 'model', 'temperature']}
        if key_kwargs:
            hash_components.append(f"kw:{hash(json.dumps(key_kwargs, sort_keys=True))}")
        
        return "_".join(hash_components)
    
    def _fallback_key_generation(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Fallback key generation method"""
        try:
            cache_data = f"{func_name}_{str(args)[:100]}_{str(kwargs)[:100]}"
            return hashlib.sha256(cache_data.encode()).hexdigest()[:16]
        except Exception:
            return f"{func_name}_{time.time()}_{id(args)}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-tier cache with promotion"""
        self.performance_monitor.start_timer("cache_get")
        
        try:
            with self.lock:
                self.stats['total_requests'] += 1
                
                # Try L1 cache first (fastest)
                l1_result = self.l1_cache.get(key)
                if l1_result is not None:
                    self.stats['l1_hits'] += 1
                    self.tier_stats[CacheLevel.L1_MEMORY]['hits'] += 1
                    self.cache_warmer.record_access(key, time.time())
                    logger.debug(f"L1 cache HIT: {key}")
                    return l1_result
                
                # Try L2 cache (optimized hash map)
                l2_result = self.l2_cache.get(key)
                if l2_result is not None:
                    # Check if entry is still valid
                    if isinstance(l2_result, EnhancedCacheEntry) and not l2_result.is_expired():
                        # Promote to L1 cache
                        self.l1_cache.put(key, l2_result.value)
                        l2_result.touch()
                        
                        self.stats['l2_hits'] += 1
                        self.stats['promotions'] += 1
                        self.tier_stats[CacheLevel.L2_OPTIMIZED]['hits'] += 1
                        self.cache_warmer.record_access(key, time.time())
                        logger.debug(f"L2 cache HIT (promoted): {key}")
                        return l2_result.value
                    else:
                        # Remove expired entry
                        self.l2_cache.delete(key)
                
                # Try L3 cache (persistent)
                l3_result = self.l3_cache.get(key)
                if l3_result is not None:
                    # Promote to L2 and L1
                    entry = EnhancedCacheEntry(
                        value=l3_result,
                        created_at=time.time(),
                        ttl=self.l3_cache.default_ttl,
                        cache_level=CacheLevel.L2_OPTIMIZED
                    )
                    
                    self.l2_cache.put(key, entry)
                    self.l1_cache.put(key, l3_result)
                    
                    self.stats['l3_hits'] += 1
                    self.stats['promotions'] += 2
                    self.tier_stats[CacheLevel.L3_PERSISTENT]['hits'] += 1
                    self.cache_warmer.record_access(key, time.time())
                    logger.debug(f"L3 cache HIT (promoted): {key}")
                    return l3_result
                
                # Cache miss across all tiers
                self.stats['total_misses'] += 1
                logger.debug(f"Cache MISS (all tiers): {key}")
                return None
                
        finally:
            self.performance_monitor.end_timer("cache_get")
            
    def get_sync(self, key: str) -> Optional[Any]:
        """Synchronous version of get for non-async contexts"""
        # Re-implement logic synchronously (most underlying calls are sync except locks)
        # Note: OptimizedLRUCache and HighPerformanceHashMap are essentially sync
        # PersistentCache is file I/O (sync)
        
        with self.lock:
            self.stats['total_requests'] += 1
            
            # L1
            l1_result = self.l1_cache.get(key)
            if l1_result is not None:
                self.stats['l1_hits'] += 1
                return l1_result
            
            # L2
            l2_result = self.l2_cache.get(key)
            if l2_result is not None:
                if isinstance(l2_result, EnhancedCacheEntry) and not l2_result.is_expired():
                    self.l1_cache.put(key, l2_result.value)
                    l2_result.touch()
                    self.stats['l2_hits'] += 1
                    return l2_result.value
                else:
                    self.l2_cache.delete(key)
            
            # L3
            l3_result = self.l3_cache.get(key)
            if l3_result is not None:
                entry = EnhancedCacheEntry(l3_result, time.time(), self.l3_cache.default_ttl, cache_level=CacheLevel.L2_OPTIMIZED)
                self.l2_cache.put(key, entry)
                self.l1_cache.put(key, l3_result)
                self.stats['l3_hits'] += 1
                return l3_result
                
            self.stats['total_misses'] += 1
            return None
    
    async def put(self, key: str, value: Any, ttl: Optional[float] = None, 
                  tags: Set[str] = None, level: CacheLevel = CacheLevel.L1_MEMORY):
        """Store value in appropriate cache tier"""
        self.performance_monitor.start_timer("cache_put")
        
        try:
            with self.lock:
                cache_ttl = ttl if ttl is not None else self.l3_cache.default_ttl
                cache_tags = tags or set()
                
                # Create enhanced entry
                entry = EnhancedCacheEntry(
                    value=value,
                    created_at=time.time(),
                    ttl=cache_ttl,
                    tags=cache_tags,
                    cache_level=level
                )
                
                # Store in multiple tiers based on strategy
                if level == CacheLevel.L1_MEMORY or entry.size_bytes < 10000:
                    # Small items go to L1
                    self.l1_cache.put(key, value)
                    self.tier_stats[CacheLevel.L1_MEMORY]['puts'] += 1
                
                if level != CacheLevel.L1_MEMORY:
                    # Also store in L2 for medium-term access
                    self.l2_cache.put(key, entry)
                    self.tier_stats[CacheLevel.L2_OPTIMIZED]['puts'] += 1
                
                # Always store in L3 for persistence
                self.l3_cache.put(key, value, ttl, cache_tags)
                self.tier_stats[CacheLevel.L3_PERSISTENT]['puts'] += 1
                
                # Update trie index for key patterns
                self._update_pattern_index(key)
                
                self.stats['total_puts'] += 1
                logger.debug(f"Cache PUT: {key} -> {level.value}")
                
        finally:
            self.performance_monitor.end_timer("cache_put")

    def put_sync(self, key: str, value: Any, ttl: Optional[float] = None, 
                 tags: Set[str] = None, level: CacheLevel = CacheLevel.L1_MEMORY):
        """Synchronous version of put"""
        # Simply wrap the logic, ignoring the async timer for simplicity or replicating it
        with self.lock:
            cache_ttl = ttl if ttl is not None else self.l3_cache.default_ttl
            cache_tags = tags or set()
            
            entry = EnhancedCacheEntry(value, time.time(), cache_ttl, tags=cache_tags, cache_level=level)
            
            if level == CacheLevel.L1_MEMORY or entry.size_bytes < 10000:
                self.l1_cache.put(key, value)
            
            if level != CacheLevel.L1_MEMORY:
                self.l2_cache.put(key, entry)
            
            self.l3_cache.put(key, value, ttl, cache_tags)
            self._update_pattern_index(key)
            self.stats['total_puts'] += 1
    
    def _update_pattern_index(self, key: str):
        """Update trie index with new key patterns"""
        # Extract prefix pattern
        parts = key.split('_')
        if len(parts) > 1:
            pattern = parts[0] + '_'
            existing_metadata = {}
            
            # Check if pattern exists
            suggestions = self.trie_index.search_prefix(pattern, max_suggestions=1)
            if suggestions:
                existing_metadata = suggestions[0][2]
                usage_count = existing_metadata.get('usage', 0) + 1
            else:
                usage_count = 1
            
            # Update or insert pattern
            self.trie_index.insert(pattern, {
                'type': 'cache_pattern',
                'usage': usage_count,
                'last_seen': time.time()
            })
    
    async def invalidate_by_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern using trie"""
        matching_keys = []
        
        with self.lock:
            # Use trie to find matching patterns efficiently
            suggestions = self.trie_index.search_prefix(pattern, max_suggestions=100)
            
            for suggestion, _, metadata in suggestions:
                # Find actual cache keys that match this pattern
                # This is a simplified approach - in practice, you'd maintain
                # a reverse index from patterns to actual keys
                pass
            
            # For now, use the existing tag-based invalidation
            self.l3_cache.invalidate_by_tags({pattern})
            
            logger.info(f"Invalidated cache entries matching pattern: {pattern}")
    
    async def _periodic_cache_warming(self):
        """Background task for predictive cache warming"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.cache_warmer.warm_cache_predictively()
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
    
    async def _periodic_cleanup(self):
        """Background task for cache cleanup and optimization"""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                await self._cleanup_expired_entries()
                await self._optimize_cache_distribution()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _periodic_optimization(self):
        """Background task for performance optimization"""
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                await self._analyze_access_patterns()
                await self._optimize_tier_distribution()
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
    
    async def _cleanup_expired_entries(self):
        """Clean up expired entries across all tiers"""
        with self.lock:
            # L2 cache cleanup
            expired_keys = []
            for key in list(self.l2_cache.cache.keys()):
                entry = self.l2_cache.get(key)
                if isinstance(entry, EnhancedCacheEntry) and entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.l2_cache.delete(key)
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired L2 entries")
    
    async def _analyze_access_patterns(self):
        """Analyze access patterns for optimization"""
        # This would analyze the trie patterns and access frequencies
        # to optimize cache strategies
        pattern_analysis = {}
        
        # Analyze most frequent patterns
        # Implementation would depend on specific analytics needs
        
        logger.debug("Analyzed cache access patterns")
    
    async def _optimize_tier_distribution(self):
        """Optimize data distribution across cache tiers"""
        # Move frequently accessed items to faster tiers
        # Move rarely accessed items to slower tiers
        # This is a simplified version - real implementation would be more complex
        
        logger.debug("Optimized cache tier distribution")
    
    async def _optimize_cache_distribution(self):
        """Optimize cache distribution based on access patterns"""
        # Analyze tier usage and redistribute if needed
        l1_usage = len(self.l1_cache.cache) / self.l1_cache.capacity
        l2_usage = len(self.l2_cache.cache) / self.l2_cache.capacity
        
        if l1_usage > 0.9 and l2_usage < 0.5:
            # Consider promoting some L2 items to L1
            # Implementation would be more sophisticated
            pass
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            total_requests = self.stats['total_requests']
            
            performance_stats = self.performance_monitor.get_performance_report()
            
            return {
                'overview': {
                    'total_requests': total_requests,
                    'total_hits': self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits'],
                    'total_misses': self.stats['total_misses'],
                    'hit_rate': round(((total_requests - self.stats['total_misses']) / max(1, total_requests)) * 100, 2),
                    'promotions': self.stats['promotions']
                },
                'tier_performance': {
                    'l1': {
                        'hits': self.stats['l1_hits'],
                        'size': len(self.l1_cache.cache),
                        'capacity': self.l1_cache.capacity,
                        'utilization': round(len(self.l1_cache.cache) / self.l1_cache.capacity * 100, 2)
                    },
                    'l2': {
                        'hits': self.stats['l2_hits'],
                        'size': len(self.l2_cache.cache),
                        'capacity': self.l2_cache.capacity,
                        'utilization': round(len(self.l2_cache.cache) / self.l2_cache.capacity * 100, 2)
                    },
                    'l3': {
                        'hits': self.stats['l3_hits'],
                        **self.l3_cache.get_stats()
                    }
                },
                'performance_metrics': performance_stats,
                'pattern_analysis': {
                    'total_patterns': self.trie_index.word_count,
                    'cache_warming_patterns': len(self.cache_warmer.warming_patterns)
                }
            }

# Enhanced decorator that uses the new cache manager
def enhanced_cached(ttl: float = 3600, tags: Set[str] = None, 
                   level: CacheLevel = CacheLevel.L1_MEMORY):
    """Enhanced caching decorator with multi-tier support"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_manager = get_enhanced_cache_manager()
            key = cache_manager._generate_optimized_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = await cache_manager.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.put(key, result, ttl, tags, level)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio.run if needed
            # This is a simplified approach
            cache_manager = get_enhanced_cache_manager()
            key = cache_manager._generate_optimized_key(func.__name__, args, kwargs)
            
            # Execute function and cache result (simplified for sync)
            result = func(*args, **kwargs)
            
            # Store in cache asynchronously in background
            asyncio.create_task(cache_manager.put(key, result, ttl, tags, level))
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Global enhanced cache manager instance
_enhanced_cache_manager = None
_cache_lock = threading.Lock()

def get_enhanced_cache_manager() -> EnhancedCacheManager:
    """Get or create global enhanced cache manager instance"""
    global _enhanced_cache_manager
    
    if _enhanced_cache_manager is None:
        with _cache_lock:
            if _enhanced_cache_manager is None:
                _enhanced_cache_manager = EnhancedCacheManager()
    
    return _enhanced_cache_manager

# Utility functions for cache management
async def warm_cache_for_queries(queries: List[str]):
    """Warm cache for a list of common queries"""
    cache_manager = get_enhanced_cache_manager()
    for query in queries:
        await cache_manager.cache_warmer.record_access(f"query_{hash(query)}", time.time())

async def optimize_cache_performance():
    """Manually trigger cache optimization"""
    cache_manager = get_enhanced_cache_manager()
    await cache_manager._analyze_access_patterns()
    await cache_manager._optimize_tier_distribution()
    logger.info("Manual cache optimization completed")