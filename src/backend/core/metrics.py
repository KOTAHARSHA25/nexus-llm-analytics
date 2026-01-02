"""
Prometheus Metrics Module for Nexus LLM Analytics
=================================================

Phase 3.8: System monitoring and observability using Prometheus metrics.

Provides:
- Request counters and latency histograms
- Agent execution metrics
- Cache performance metrics
- LLM token usage tracking
- Error rate monitoring

Usage:
    from src.backend.core.metrics import track_request, update_cache_metrics, METRICS
    
    @track_request(endpoint="query")
    async def handle_query(...):
        ...

Version: 1.0.0
"""

import time
import logging
from functools import wraps
from typing import Dict, Any, Callable, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Check if prometheus_client is available
PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not installed. Metrics will use fallback implementation.")
    # Define dummy classes for when prometheus is not installed
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'unknown')
            self._value = 0
            self._labels = {}
        
        def labels(self, *args, **kwargs):
            return self
        
        def inc(self, amount=1):
            self._value += amount
        
        def dec(self, amount=1):
            self._value -= amount
        
        def set(self, value):
            self._value = value
        
        def observe(self, value):
            pass
        
        @contextmanager
        def time(self):
            start = time.time()
            yield
            pass
        
        def info(self, info_dict):
            pass


class MetricsRegistry:
    """
    Central registry for all system metrics.
    
    Provides both Prometheus metrics (when available) and fallback
    in-memory metrics for systems without prometheus_client.
    """
    
    def __init__(self):
        self._fallback_metrics: Dict[str, Any] = {
            'requests_total': 0,
            'requests_by_endpoint': {},
            'request_latencies': [],
            'errors_total': 0,
            'errors_by_type': {},
            'agent_executions': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_tokens_used': 0,
            'active_requests': 0
        }
        
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        else:
            self._init_fallback_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Request metrics
        self.requests_total = Counter(
            'nexus_requests_total',
            'Total number of requests',
            ['endpoint', 'method', 'status']
        )
        
        self.request_latency = Histogram(
            'nexus_request_latency_seconds',
            'Request latency in seconds',
            ['endpoint'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.active_requests = Gauge(
            'nexus_active_requests',
            'Number of active requests'
        )
        
        # Agent metrics
        self.agent_executions = Counter(
            'nexus_agent_executions_total',
            'Total agent executions',
            ['agent', 'status']
        )
        
        self.agent_latency = Histogram(
            'nexus_agent_latency_seconds',
            'Agent execution latency',
            ['agent'],
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        # LLM metrics
        self.llm_tokens_total = Counter(
            'nexus_llm_tokens_total',
            'Total LLM tokens used',
            ['model', 'type']  # type: input/output
        )
        
        self.llm_requests_total = Counter(
            'nexus_llm_requests_total',
            'Total LLM API requests',
            ['model', 'status']
        )
        
        self.llm_latency = Histogram(
            'nexus_llm_latency_seconds',
            'LLM request latency',
            ['model'],
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'nexus_cache_hits_total',
            'Cache hits',
            ['cache_type']
        )
        
        self.cache_misses = Counter(
            'nexus_cache_misses_total',
            'Cache misses',
            ['cache_type']
        )
        
        self.cache_size = Gauge(
            'nexus_cache_size',
            'Current cache size',
            ['cache_type']
        )
        
        # Error metrics
        self.errors_total = Counter(
            'nexus_errors_total',
            'Total errors',
            ['error_type', 'endpoint']
        )
        
        # RAG metrics (Phase 3 specific)
        self.rag_queries_total = Counter(
            'nexus_rag_queries_total',
            'Total RAG queries',
            ['search_type']  # hybrid, vector, keyword
        )
        
        self.rag_latency = Histogram(
            'nexus_rag_latency_seconds',
            'RAG query latency',
            ['search_type'],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
        )
        
        self.documents_indexed = Gauge(
            'nexus_documents_indexed',
            'Number of indexed documents'
        )
        
        # System info
        self.system_info = Info(
            'nexus_system',
            'System information'
        )
        self.system_info.info({
            'version': '3.0.0',
            'phase': '3'
        })
        
        logger.info("Prometheus metrics initialized")
    
    def _init_fallback_metrics(self):
        """Initialize fallback metrics (when prometheus not available)"""
        self.requests_total = DummyMetric(name='requests_total')
        self.request_latency = DummyMetric(name='request_latency')
        self.active_requests = DummyMetric(name='active_requests')
        self.agent_executions = DummyMetric(name='agent_executions')
        self.agent_latency = DummyMetric(name='agent_latency')
        self.llm_tokens_total = DummyMetric(name='llm_tokens')
        self.llm_requests_total = DummyMetric(name='llm_requests')
        self.llm_latency = DummyMetric(name='llm_latency')
        self.cache_hits = DummyMetric(name='cache_hits')
        self.cache_misses = DummyMetric(name='cache_misses')
        self.cache_size = DummyMetric(name='cache_size')
        self.errors_total = DummyMetric(name='errors')
        self.rag_queries_total = DummyMetric(name='rag_queries')
        self.rag_latency = DummyMetric(name='rag_latency')
        self.documents_indexed = DummyMetric(name='documents_indexed')
        self.system_info = DummyMetric(name='system_info')
        
        logger.info("Fallback metrics initialized (prometheus_client not available)")
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get statistics from fallback metrics"""
        return self._fallback_metrics.copy()


# Global metrics registry - singleton pattern
_METRICS_INSTANCE = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry (singleton)"""
    global _METRICS_INSTANCE
    if _METRICS_INSTANCE is None:
        _METRICS_INSTANCE = MetricsRegistry()
    return _METRICS_INSTANCE


def reset_metrics_registry():
    """Reset the metrics registry (for testing)"""
    global _METRICS_INSTANCE
    if PROMETHEUS_AVAILABLE:
        # Unregister all collectors to avoid duplicates
        from prometheus_client import REGISTRY
        collectors_to_remove = []
        for collector in REGISTRY._names_to_collectors.values():
            collectors_to_remove.append(collector)
        for collector in set(collectors_to_remove):
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass
    _METRICS_INSTANCE = None


# For backwards compatibility
METRICS = get_metrics_registry()


def track_request(endpoint: str = "unknown"):
    """
    Decorator to track request metrics.
    
    Args:
        endpoint: Name of the endpoint being tracked
    
    Example:
        @track_request(endpoint="query")
        async def handle_query(request):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            METRICS.active_requests.inc()
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                METRICS.errors_total.labels(
                    error_type=type(e).__name__,
                    endpoint=endpoint
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                METRICS.active_requests.dec()
                METRICS.requests_total.labels(
                    endpoint=endpoint,
                    method="POST",
                    status=status
                ).inc()
                METRICS.request_latency.labels(endpoint=endpoint).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            METRICS.active_requests.inc()
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                METRICS.errors_total.labels(
                    error_type=type(e).__name__,
                    endpoint=endpoint
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                METRICS.active_requests.dec()
                METRICS.requests_total.labels(
                    endpoint=endpoint,
                    method="POST",
                    status=status
                ).inc()
                METRICS.request_latency.labels(endpoint=endpoint).observe(duration)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def track_agent_execution(agent_name: str):
    """
    Decorator to track agent execution metrics.
    
    Args:
        agent_name: Name of the agent
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                METRICS.agent_executions.labels(agent=agent_name, status=status).inc()
                METRICS.agent_latency.labels(agent=agent_name).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                METRICS.agent_executions.labels(agent=agent_name, status=status).inc()
                METRICS.agent_latency.labels(agent=agent_name).observe(duration)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def track_llm_usage(model: str, input_tokens: int, output_tokens: int, latency: float, success: bool = True):
    """
    Track LLM token usage and performance.
    
    Args:
        model: Model name (e.g., "gpt-4", "gemini-pro")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency: Request latency in seconds
        success: Whether the request was successful
    """
    METRICS.llm_tokens_total.labels(model=model, type="input").inc(input_tokens)
    METRICS.llm_tokens_total.labels(model=model, type="output").inc(output_tokens)
    METRICS.llm_requests_total.labels(model=model, status="success" if success else "error").inc()
    METRICS.llm_latency.labels(model=model).observe(latency)


def update_cache_metrics(cache_type: str, hits: int = 0, misses: int = 0, size: Optional[int] = None):
    """
    Update cache-related metrics.
    
    Args:
        cache_type: Type of cache (query, model, file_analysis, semantic)
        hits: Number of hits to add
        misses: Number of misses to add
        size: Current cache size (optional)
    """
    if hits > 0:
        METRICS.cache_hits.labels(cache_type=cache_type).inc(hits)
    if misses > 0:
        METRICS.cache_misses.labels(cache_type=cache_type).inc(misses)
    if size is not None:
        METRICS.cache_size.labels(cache_type=cache_type).set(size)


def track_rag_query(search_type: str = "hybrid"):
    """
    Decorator to track RAG query metrics.
    
    Args:
        search_type: Type of search (hybrid, vector, keyword)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                METRICS.rag_queries_total.labels(search_type=search_type).inc()
                METRICS.rag_latency.labels(search_type=search_type).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                METRICS.rag_queries_total.labels(search_type=search_type).inc()
                METRICS.rag_latency.labels(search_type=search_type).observe(duration)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


@contextmanager
def time_operation(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Context manager for timing operations.
    
    Args:
        metric_name: Name of the metric to update
        labels: Optional labels for the metric
    
    Example:
        with time_operation("custom_operation"):
            do_something()
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        # Log the timing for custom metrics
        logger.debug(f"Operation '{metric_name}' took {duration:.3f}s")


def generate_metrics_output() -> bytes:
    """
    Generate Prometheus metrics output.
    
    Returns:
        Prometheus metrics in text format
    """
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY)
    else:
        # Return a basic text format for fallback metrics
        output_lines = [
            "# Nexus LLM Analytics Metrics (Fallback Mode)",
            "# prometheus_client not installed - using in-memory metrics",
            "",
            f"nexus_requests_total {METRICS._fallback_metrics['requests_total']}",
            f"nexus_errors_total {METRICS._fallback_metrics['errors_total']}",
            f"nexus_cache_hits {METRICS._fallback_metrics['cache_hits']}",
            f"nexus_cache_misses {METRICS._fallback_metrics['cache_misses']}",
            f"nexus_llm_tokens_used {METRICS._fallback_metrics['llm_tokens_used']}",
            f"nexus_active_requests {METRICS._fallback_metrics['active_requests']}",
        ]
        return "\n".join(output_lines).encode('utf-8')


def get_metrics_content_type() -> str:
    """Get the content type for metrics response"""
    if PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return "text/plain; charset=utf-8"


# Export commonly used items
__all__ = [
    'METRICS',
    'track_request',
    'track_agent_execution', 
    'track_llm_usage',
    'update_cache_metrics',
    'track_rag_query',
    'time_operation',
    'generate_metrics_output',
    'get_metrics_content_type',
    'PROMETHEUS_AVAILABLE'
]
