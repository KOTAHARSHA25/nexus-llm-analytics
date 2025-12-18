[{
	"resource": "/c:/Users/KOTA/Downloads/nexus-llm-analytics/src/frontend/app/page.tsx",
	"owner": "typescript",
	"code": "2322",
	"severity": 8,
	"message": "Type 'FileInfo[]' is not assignable to type 'import(\"c:/Users/KOTA/Downloads/nexus-llm-analytics/src/frontend/hooks/useDashboardState\").FileInfo[]'.\n  Type 'FileInfo' is missing the following properties from type 'FileInfo': id, uploadedAt",
	"source": "ts",
	"startLineNumber": 429,
	"startColumn": 69,
	"endLineNumber": 429,
	"endColumn": 82,
	"relatedInformation": [
		{
			"startLineNumber": 32,
			"startColumn": 3,
			"endLineNumber": 32,
			"endColumn": 16,
			"message": "The expected type comes from property 'uploadedFiles' which is declared here on type 'IntrinsicAttributes & FileUploadProps'",
			"resource": "/c:/Users/KOTA/Downloads/nexus-llm-analytics/src/frontend/components/file-upload.tsx"
		}
	],
	"origin": "extHost1"
}]# Unit Tests for LLM Client
# Production-grade unit testing for optimized LLM client

import pytest
import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp
from aiohttp import ClientSession, ClientResponse, ClientError
import sys
import os

# Import the optimized LLM client
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_llm_client import (
    OptimizedLLMClient,
    SmartCache,
    OptimizedConnectionPool as ConnectionPool
)
from backend.core.rate_limiter import RateLimiter

# Mock implementations for missing classes
class ResponseStreamer:
    def stream(self, data):
        return iter([data])

class LLMProvider:
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class QueryType:
    ANALYSIS = "analysis"
    CHAT = "chat"

class ProcessingPriority:
    HIGH = "high"
    NORMAL = "normal"

class TestSmartCache:
    """Unit tests for SmartCache component"""
    
    @pytest.fixture
    def cache(self):
        """Create fresh cache instance"""
        return SmartCache(max_size=100, ttl=3600)
    
    @pytest.fixture
    def populated_cache(self):
        """Create cache with sample data"""
        cache = SmartCache(max_size=100, ttl=3600)
        for i in range(50):
            cache.put(f"key_{i}", f"value_{i}")
        return cache
    
    @pytest.mark.unit
    def test_cache_initialization(self, cache):
        """Test cache proper initialization"""
        assert cache.max_size == 100
        assert cache.ttl == 3600
        assert cache.size == 0
        assert cache.is_empty()
    
    @pytest.mark.unit
    def test_basic_put_get_operations(self, cache):
        """Test basic cache operations"""
        # Test put and get
        cache.put("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        assert cache.size == 1
        assert not cache.is_empty()
    
    @pytest.mark.unit
    def test_cache_hit_miss_tracking(self, cache):
        """Test cache hit/miss statistics"""
        # Initial state
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        
        # Add item and test hit
        cache.put("key1", "value1")
        hit_result = cache.get("key1")
        assert hit_result == "value1"
        
        # Test miss
        miss_result = cache.get("nonexistent")
        assert miss_result is None
        
        # Check updated stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    @pytest.mark.unit
    def test_ttl_expiration(self, cache):
        """Test TTL-based expiration"""
        # Create cache with short TTL
        short_ttl_cache = SmartCache(max_size=10, ttl=0.1)  # 100ms TTL
        
        short_ttl_cache.put("expiring_key", "expiring_value")
        
        # Should be available immediately
        assert short_ttl_cache.get("expiring_key") == "expiring_value"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired now
        assert short_ttl_cache.get("expiring_key") is None
    
    @pytest.mark.unit
    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        # Create small cache to force eviction
        small_cache = SmartCache(max_size=3, ttl=3600)
        
        # Fill cache to capacity
        small_cache.put("key1", "value1")
        small_cache.put("key2", "value2")
        small_cache.put("key3", "value3")
        
        assert small_cache.size == 3
        
        # Access key1 to make it recently used
        small_cache.get("key1")
        
        # Add new item, should evict key2 (least recently used)
        small_cache.put("key4", "value4")
        
        assert small_cache.size == 3
        assert small_cache.get("key1") == "value1"  # Still present
        assert small_cache.get("key2") is None      # Evicted
        assert small_cache.get("key3") == "value3"  # Still present
        assert small_cache.get("key4") == "value4"  # Newly added
    
    @pytest.mark.unit
    def test_cache_key_hashing(self, cache):
        """Test cache key hashing for complex objects"""
        # Test with different key types
        test_keys = [
            "simple_string",
            ("tuple", "key"),
            {"dict": "key", "nested": {"value": 123}},
            ["list", "key", 456]
        ]
        
        for i, key in enumerate(test_keys):
            cache.put(key, f"value_{i}")
        
        # Verify all keys work
        for i, key in enumerate(test_keys):
            assert cache.get(key) == f"value_{i}"
    
    @pytest.mark.unit
    def test_cache_clear(self, populated_cache):
        """Test cache clearing"""
        assert populated_cache.size > 0
        
        populated_cache.clear()
        
        assert populated_cache.size == 0
        assert populated_cache.is_empty()
        
        # Verify stats reset
        stats = populated_cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_cache_performance(self, performance_timer):
        """Test cache performance with large dataset"""
        cache = SmartCache(max_size=10000, ttl=3600)
        
        # Test insertion performance
        performance_timer.start()
        for i in range(5000):
            cache.put(f"perf_key_{i}", f"perf_value_{i}")
        performance_timer.stop()
        
        insertion_time = performance_timer.elapsed
        insertions_per_second = 5000 / insertion_time
        
        assert insertions_per_second > 10000, f"Only {insertions_per_second:.0f} insertions/sec"
        
        # Test lookup performance
        performance_timer.start()
        for i in range(0, 5000, 10):  # Sample every 10th item
            value = cache.get(f"perf_key_{i}")
            assert value == f"perf_value_{i}"
        performance_timer.stop()
        
        lookup_time = performance_timer.elapsed
        lookups_per_second = 500 / lookup_time
        
        assert lookups_per_second > 50000, f"Only {lookups_per_second:.0f} lookups/sec"


class TestConnectionPool:
    """Unit tests for ConnectionPool component"""
    
    @pytest.fixture
    def connection_pool(self):
        """Create fresh connection pool"""
        return ConnectionPool(max_connections=5, timeout=30.0)
    
    @pytest.mark.unit
    def test_pool_initialization(self, connection_pool):
        """Test connection pool initialization"""
        assert connection_pool.max_connections == 5
        assert connection_pool.timeout == 30.0
        assert connection_pool.active_connections == 0
    
    @pytest.mark.unit
    async def test_connection_acquisition_release(self, connection_pool):
        """Test connection acquisition and release"""
        # Mock session
        mock_session = AsyncMock(spec=ClientSession)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Acquire connection
            session = await connection_pool.acquire()
            assert session is not None
            assert connection_pool.active_connections == 1
            
            # Release connection
            await connection_pool.release(session)
            assert connection_pool.active_connections == 0
    
    @pytest.mark.unit
    async def test_connection_pool_limits(self, connection_pool):
        """Test connection pool respects limits"""
        sessions = []
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value = AsyncMock(spec=ClientSession)
            
            # Acquire up to max connections
            for i in range(connection_pool.max_connections):
                session = await connection_pool.acquire()
                sessions.append(session)
            
            assert connection_pool.active_connections == connection_pool.max_connections
            
            # Attempting to acquire more should timeout or wait
            with pytest.raises((asyncio.TimeoutError, Exception)):
                await asyncio.wait_for(connection_pool.acquire(), timeout=0.1)
            
            # Release connections
            for session in sessions:
                await connection_pool.release(session)
    
    @pytest.mark.unit
    async def test_connection_health_check(self, connection_pool):
        """Test connection health checking"""
        # Mock unhealthy session
        unhealthy_session = AsyncMock(spec=ClientSession)
        unhealthy_session.closed = True
        
        # Mock healthy session
        healthy_session = AsyncMock(spec=ClientSession)
        healthy_session.closed = False
        
        with patch('aiohttp.ClientSession', return_value=healthy_session):
            # Should replace unhealthy connection
            is_healthy = await connection_pool.is_connection_healthy(unhealthy_session)
            assert not is_healthy
            
            is_healthy = await connection_pool.is_connection_healthy(healthy_session)
            assert is_healthy
    
    @pytest.mark.unit
    async def test_pool_cleanup(self, connection_pool):
        """Test connection pool cleanup"""
        sessions = []
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock(spec=ClientSession)
            mock_session_class.return_value = mock_session
            
            # Acquire connections
            for i in range(3):
                session = await connection_pool.acquire()
                sessions.append(session)
            
            # Cleanup pool
            await connection_pool.cleanup()
            
            # Verify all sessions were closed
            assert mock_session.close.call_count >= 3


class TestRateLimiter:
    """Unit tests for RateLimiter component"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter with test limits"""
        return RateLimiter(requests_per_minute=60, requests_per_hour=1000)
    
    @pytest.mark.unit
    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initialization"""
        assert rate_limiter.requests_per_minute == 60
        assert rate_limiter.requests_per_hour == 1000
        assert rate_limiter.current_minute_count == 0
        assert rate_limiter.current_hour_count == 0
    
    @pytest.mark.unit
    async def test_rate_limiting_within_limits(self, rate_limiter):
        """Test requests within rate limits"""
        # Should allow requests within limits
        for i in range(10):
            allowed = await rate_limiter.is_request_allowed()
            assert allowed is True
            
            await rate_limiter.record_request()
        
        assert rate_limiter.current_minute_count == 10
        assert rate_limiter.current_hour_count == 10
    
    @pytest.mark.unit
    async def test_rate_limiting_exceeds_limits(self):
        """Test rate limiting when exceeding limits"""
        # Create strict rate limiter
        strict_limiter = RateLimiter(requests_per_minute=2, requests_per_hour=10)
        
        # Make requests up to limit
        for i in range(2):
            allowed = await strict_limiter.is_request_allowed()
            assert allowed is True
            await strict_limiter.record_request()
        
        # Next request should be denied
        allowed = await strict_limiter.is_request_allowed()
        assert allowed is False
    
    @pytest.mark.unit
    async def test_rate_limiter_reset(self, rate_limiter):
        """Test rate limiter reset functionality"""
        # Make some requests
        for i in range(5):
            await rate_limiter.record_request()
        
        assert rate_limiter.current_minute_count == 5
        
        # Reset counters
        rate_limiter.reset_counters()
        
        assert rate_limiter.current_minute_count == 0
        assert rate_limiter.current_hour_count == 0
    
    @pytest.mark.unit
    async def test_provider_specific_limiting(self):
        """Test provider-specific rate limiting"""
        # Different limits for different providers
        openai_limiter = RateLimiter(requests_per_minute=20, requests_per_hour=500)
        anthropic_limiter = RateLimiter(requests_per_minute=30, requests_per_hour=800)
        
        # Simulate different usage patterns
        for i in range(15):
            assert await openai_limiter.is_request_allowed()
            await openai_limiter.record_request()
        
        for i in range(25):
            assert await anthropic_limiter.is_request_allowed()
            await anthropic_limiter.record_request()
        
        # Verify different states
        assert openai_limiter.current_minute_count == 15
        assert anthropic_limiter.current_minute_count == 25


class TestResponseStreamer:
    """Unit tests for ResponseStreamer component"""
    
    @pytest.fixture
    def response_streamer(self):
        """Create response streamer"""
        return ResponseStreamer()
    
    @pytest.mark.unit
    async def test_streaming_response_processing(self, response_streamer):
        """Test streaming response processing"""
        # Mock streaming response
        mock_chunks = [
            b'{"chunk": 1, "content": "Hello"}',
            b'{"chunk": 2, "content": " world"}',
            b'{"chunk": 3, "content": "!"}',
            b'{"chunk": 4, "done": true}'
        ]
        
        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk
        
        # Process stream
        result = await response_streamer.process_stream(mock_stream())
        
        assert result is not None
        assert 'content' in result
        assert 'chunks_processed' in result
        assert result['chunks_processed'] == len(mock_chunks)
    
    @pytest.mark.unit
    async def test_stream_error_handling(self, response_streamer):
        """Test stream error handling"""
        async def failing_stream():
            yield b'{"chunk": 1, "content": "Start"}'
            raise aiohttp.ClientError("Stream failed")
        
        # Should handle stream errors gracefully
        result = await response_streamer.process_stream(failing_stream())
        
        assert result is not None
        assert 'error' in result
        assert 'partial_content' in result
    
    @pytest.mark.unit
    async def test_stream_timeout_handling(self, response_streamer):
        """Test stream timeout handling"""
        async def slow_stream():
            yield b'{"chunk": 1, "content": "Start"}'
            await asyncio.sleep(10)  # Simulate slow response
            yield b'{"chunk": 2, "content": "End"}'
        
        # Should timeout gracefully
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                response_streamer.process_stream(slow_stream()),
                timeout=0.1
            )


class TestOptimizedLLMClient:
    """Comprehensive unit tests for OptimizedLLMClient"""
    
    @pytest.fixture
    def llm_client(self):
        """Create LLM client with test configuration"""
        config = {
            'providers': {
                'openai': {
                    'api_key': 'test_key',
                    'rate_limit': {'requests_per_minute': 60}
                }
            },
            'cache': {
                'max_size': 100,
                'ttl': 3600
            },
            'connection_pool': {
                'max_connections': 5
            }
        }
        return OptimizedLLMClient(config)
    
    @pytest.fixture
    def mock_api_response(self):
        """Create mock API response"""
        return {
            'choices': [
                {
                    'message': {
                        'content': 'This is a test response from the LLM.',
                        'role': 'assistant'
                    }
                }
            ],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 8,
                'total_tokens': 18
            },
            'model': 'gpt-3.5-turbo'
        }
    
    @pytest.mark.unit
    def test_client_initialization(self, llm_client):
        """Test LLM client initialization"""
        assert llm_client is not None
        assert llm_client.cache is not None
        assert llm_client.connection_pool is not None
        assert len(llm_client.rate_limiters) > 0
    
    @pytest.mark.unit
    async def test_simple_query_processing(self, llm_client, mock_api_response):
        """Test simple query processing"""
        query = "What is the capital of France?"
        
        with patch.object(llm_client, '_make_api_request', return_value=mock_api_response):
            result = await llm_client.process_query(query)
            
            assert result is not None
            assert 'content' in result
            assert 'model' in result
            assert 'token_count' in result
            assert result['content'] == 'This is a test response from the LLM.'
    
    @pytest.mark.unit
    async def test_cache_utilization(self, llm_client, mock_api_response):
        """Test cache hit/miss behavior"""
        query = "Test query for caching"
        
        with patch.object(llm_client, '_make_api_request', return_value=mock_api_response) as mock_api:
            # First request - should hit API
            result1 = await llm_client.process_query(query)
            assert mock_api.call_count == 1
            
            # Second identical request - should hit cache
            result2 = await llm_client.process_query(query)
            assert mock_api.call_count == 1  # No additional API call
            
            # Results should be identical
            assert result1['content'] == result2['content']
            assert result2.get('from_cache') is True
    
    @pytest.mark.unit
    async def test_batch_processing(self, llm_client, mock_api_response):
        """Test batch query processing"""
        queries = [
            "Query 1: What is AI?",
            "Query 2: How does machine learning work?",
            "Query 3: Explain neural networks"
        ]
        
        with patch.object(llm_client, '_make_api_request', return_value=mock_api_response):
            results = await llm_client.process_batch(queries)
            
            assert len(results) == len(queries)
            for result in results:
                assert 'content' in result
                assert 'processing_time' in result
    
    @pytest.mark.unit
    async def test_rate_limiting_enforcement(self, llm_client):
        """Test rate limiting enforcement"""
        # Create client with strict rate limits
        strict_config = {
            'providers': {
                'openai': {
                    'api_key': 'test_key',
                    'rate_limit': {'requests_per_minute': 2}
                }
            }
        }
        strict_client = OptimizedLLMClient(strict_config)
        
        # Make requests up to limit
        with patch.object(strict_client, '_make_api_request', return_value={'content': 'test'}):
            # First two requests should succeed
            result1 = await strict_client.process_query("Query 1")
            result2 = await strict_client.process_query("Query 2")
            
            assert result1 is not None
            assert result2 is not None
            
            # Third request should be rate limited
            result3 = await strict_client.process_query("Query 3")
            assert 'rate_limited' in result3 or 'error' in result3
    
    @pytest.mark.unit
    async def test_provider_failover(self, llm_client):
        """Test provider failover mechanism"""
        query = "Test failover query"
        
        # Mock primary provider failure
        def mock_api_request_with_failover(provider, *args, **kwargs):
            if provider == LLMProvider.OPENAI:
                raise ClientError("OpenAI API failed")
            return {'content': 'Failover response', 'model': 'claude-3'}
        
        with patch.object(llm_client, '_make_api_request', side_effect=mock_api_request_with_failover):
            result = await llm_client.process_query(query)
            
            assert result is not None
            assert result['content'] == 'Failover response'
            assert 'failover_used' in result
    
    @pytest.mark.unit
    async def test_streaming_response_handling(self, llm_client):
        """Test streaming response handling"""
        query = "Generate a long response"
        
        # Mock streaming response
        async def mock_streaming_request(*args, **kwargs):
            return {
                'content': 'Streaming response content',
                'streaming': True,
                'chunks_processed': 5
            }
        
        with patch.object(llm_client, '_make_streaming_request', side_effect=mock_streaming_request):
            result = await llm_client.process_query(query, stream=True)
            
            assert result is not None
            assert result['streaming'] is True
            assert 'chunks_processed' in result
    
    @pytest.mark.unit
    async def test_query_optimization(self, llm_client):
        """Test intelligent query optimization"""
        # Test different query types
        queries = [
            ("Simple question", QueryType.SIMPLE_QA),
            ("Analyze this data and provide insights", QueryType.DATA_ANALYSIS),
            ("Generate a creative story", QueryType.CREATIVE_GENERATION),
            ("Summarize this long document", QueryType.SUMMARIZATION)
        ]
        
        for query, expected_type in queries:
            optimization = llm_client._optimize_query(query)
            
            assert optimization is not None
            assert 'query_type' in optimization
            assert 'priority' in optimization
            assert 'estimated_tokens' in optimization
    
    @pytest.mark.unit
    async def test_error_handling_and_retry(self, llm_client):
        """Test error handling and retry logic"""
        query = "Test error handling"
        
        # Mock API that fails twice then succeeds
        call_count = 0
        def mock_api_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ClientError(f"API error attempt {call_count}")
            return {'content': 'Success after retry', 'model': 'gpt-3.5-turbo'}
        
        with patch.object(llm_client, '_make_api_request', side_effect=mock_api_with_retry):
            result = await llm_client.process_query(query)
            
            assert result is not None
            assert result['content'] == 'Success after retry'
            assert call_count == 3  # Failed 2 times, succeeded on 3rd
    
    @pytest.mark.unit 
    @pytest.mark.performance
    async def test_concurrent_request_handling(self, llm_client, performance_timer):
        """Test concurrent request handling performance"""
        queries = [f"Concurrent query {i}" for i in range(20)]
        
        with patch.object(llm_client, '_make_api_request', 
                         return_value={'content': 'Response', 'model': 'test'}):
            
            performance_timer.start()
            
            # Process queries concurrently
            tasks = [llm_client.process_query(query) for query in queries]
            results = await asyncio.gather(*tasks)
            
            performance_timer.stop()
            
            # Verify results
            assert len(results) == len(queries)
            for result in results:
                assert 'content' in result
            
            # Check performance
            total_time = performance_timer.elapsed
            queries_per_second = len(queries) / total_time
            
            assert queries_per_second > 10, f"Only {queries_per_second:.1f} queries/sec"
    
    @pytest.mark.unit
    async def test_memory_management(self, llm_client):
        """Test memory management during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many queries
        queries = [f"Memory test query {i}" for i in range(100)]
        
        with patch.object(llm_client, '_make_api_request', 
                         return_value={'content': 'Test response', 'model': 'test'}):
            
            for query in queries:
                result = await llm_client.process_query(query)
                del result  # Explicit cleanup
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
    
    @pytest.mark.unit
    async def test_performance_metrics_collection(self, llm_client):
        """Test performance metrics collection"""
        # Process some queries
        with patch.object(llm_client, '_make_api_request', 
                         return_value={'content': 'Test', 'model': 'test'}):
            
            for i in range(10):
                await llm_client.process_query(f"Test query {i}")
        
        # Get performance metrics
        metrics = llm_client.get_performance_metrics()
        
        assert metrics is not None
        assert 'total_requests' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'average_response_time' in metrics
        assert 'provider_usage' in metrics
        
        assert metrics['total_requests'] == 10
        assert 0 <= metrics['cache_hit_rate'] <= 1
    
    @pytest.mark.unit
    async def test_configuration_validation(self):
        """Test configuration validation"""
        # Test invalid configuration
        invalid_configs = [
            {},  # Empty config
            {'providers': {}},  # No providers
            {'providers': {'openai': {}}},  # Missing API key
        ]
        
        for config in invalid_configs:
            with pytest.raises((ValueError, KeyError)):
                OptimizedLLMClient(config)
        
        # Test valid configuration
        valid_config = {
            'providers': {
                'openai': {
                    'api_key': 'test_key',
                    'rate_limit': {'requests_per_minute': 60}
                }
            }
        }
        
        client = OptimizedLLMClient(valid_config)
        assert client is not None
    
    @pytest.mark.unit
    async def test_cleanup_and_resource_management(self, llm_client):
        """Test proper cleanup and resource management"""
        # Process some queries to create resources
        with patch.object(llm_client, '_make_api_request', 
                         return_value={'content': 'Test', 'model': 'test'}):
            
            for i in range(5):
                await llm_client.process_query(f"Query {i}")
        
        # Test cleanup
        await llm_client.cleanup()
        
        # Verify resources were cleaned up
        assert llm_client.connection_pool.active_connections == 0


# Integration tests combining multiple LLM components
class TestLLMClientIntegration:
    """Integration tests for LLM client components"""
    
    @pytest.mark.integration
    async def test_full_workflow_integration(self):
        """Test complete LLM workflow integration"""
        config = {
            'providers': {
                'openai': {
                    'api_key': 'test_key',
                    'rate_limit': {'requests_per_minute': 60}
                }
            },
            'cache': {'max_size': 50, 'ttl': 1800},
            'connection_pool': {'max_connections': 3}
        }
        
        client = OptimizedLLMClient(config)
        
        # Mock successful API response
        mock_response = {
            'choices': [{'message': {'content': 'Integration test response'}}],
            'usage': {'total_tokens': 20},
            'model': 'gpt-3.5-turbo'
        }
        
        with patch.object(client, '_make_api_request', return_value=mock_response):
            # Test single query
            result = await client.process_query("Integration test query")
            assert result['content'] == 'Integration test response'
            
            # Test caching
            cached_result = await client.process_query("Integration test query")
            assert cached_result['from_cache'] is True
            
            # Test batch processing
            batch_queries = ["Query 1", "Query 2", "Query 3"]
            batch_results = await client.process_batch(batch_queries)
            assert len(batch_results) == 3
            
            # Test metrics
            metrics = client.get_performance_metrics()
            assert metrics['total_requests'] > 0
        
        await client.cleanup()
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_performance_under_load(self, performance_timer):
        """Test system performance under load"""
        config = {
            'providers': {
                'openai': {
                    'api_key': 'test_key',
                    'rate_limit': {'requests_per_minute': 1000}
                }
            },
            'cache': {'max_size': 1000, 'ttl': 3600},
            'connection_pool': {'max_connections': 10}
        }
        
        client = OptimizedLLMClient(config)
        
        # Generate load
        num_queries = 100
        queries = [f"Load test query {i}" for i in range(num_queries)]
        
        mock_response = {
            'choices': [{'message': {'content': f'Response'}}],
            'usage': {'total_tokens': 15},
            'model': 'gpt-3.5-turbo'
        }
        
        with patch.object(client, '_make_api_request', return_value=mock_response):
            performance_timer.start()
            
            # Process queries in batches
            batch_size = 10
            for i in range(0, num_queries, batch_size):
                batch = queries[i:i+batch_size]
                await client.process_batch(batch)
            
            performance_timer.stop()
        
        total_time = performance_timer.elapsed
        throughput = num_queries / total_time
        
        assert throughput > 50, f"Throughput {throughput:.1f} queries/sec too low"
        
        # Verify system stability
        metrics = client.get_performance_metrics()
        assert metrics['total_requests'] == num_queries
        
        await client.cleanup()


if __name__ == '__main__':
    # Run specific test categories
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--cov=src.backend.core.optimized_llm_client',
        '--cov-report=term-missing'
    ])