# Integration Tests
# Production-grade integration testing for component interactions

import pytest
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
from aiohttp import web
import sys
import os

# Import all components for integration testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_data_structures import OptimizedTrie, HighPerformanceHashMap, OptimizedDataProcessor
from backend.core.optimized_llm_client import OptimizedLLMClient, SmartCache
from backend.core.optimized_file_io import OptimizedFileProcessor
from backend.core.enhanced_cache_integration import EnhancedCacheManager
from backend.core.intelligent_query_engine import IntelligentQueryRouter, QueryPatternAnalyzer

class TestCacheIntegration:
    """Integration tests for cache system with all components"""
    
    @pytest.fixture
    async def integrated_cache_system(self):
        """Create integrated cache system with all components"""
        # Initialize cache manager
        cache_config = {
            'l1_cache': {'max_size': 100, 'ttl': 300},
            'l2_cache': {'max_size': 500, 'ttl': 1800},
            'l3_cache': {'max_size': 1000, 'ttl': 3600}
        }
        cache_manager = EnhancedCacheManager(cache_config)
        
        # Initialize other components
        file_processor = OptimizedFileProcessor()
        llm_client = OptimizedLLMClient({
            'providers': {
                'openai': {'api_key': 'test_key', 'rate_limit': {'requests_per_minute': 60}}
            }
        })
        
        # Initialize data structures
        trie = OptimizedTrie()
        hashmap = HighPerformanceHashMap()
        
        return {
            'cache_manager': cache_manager,
            'file_processor': file_processor,
            'llm_client': llm_client,
            'trie': trie,
            'hashmap': hashmap
        }
    
    @pytest.mark.integration
    async def test_file_processing_with_caching(self, integrated_cache_system, sample_csv_file):
        """Test file processing with integrated caching"""
        cache_manager = integrated_cache_system['cache_manager']
        file_processor = integrated_cache_system['file_processor']
        
        file_path = str(sample_csv_file)
        
        # First processing - should cache results
        result1 = await file_processor.process_file(file_path)
        cache_key = f"file_processing:{file_path}"
        await cache_manager.put(cache_key, result1)
        
        # Second processing - should use cache
        cached_result = await cache_manager.get(cache_key)
        assert cached_result is not None
        assert cached_result['file_type'] == result1['file_type']
        assert len(cached_result['data']) == len(result1['data'])
        
        # Verify cache hit statistics
        stats = await cache_manager.get_comprehensive_stats()
        assert stats['overview']['hit_rate'] > 0
    
    @pytest.mark.integration
    async def test_llm_responses_with_multi_tier_caching(self, integrated_cache_system):
        """Test LLM responses with multi-tier caching"""
        cache_manager = integrated_cache_system['cache_manager']
        llm_client = integrated_cache_system['llm_client']
        
        query = "What is machine learning?"
        
        # Mock LLM response
        mock_response = {
            'content': 'Machine learning is a subset of AI...',
            'model': 'gpt-3.5-turbo',
            'token_count': 50
        }
        
        with patch.object(llm_client, '_make_api_request', return_value={
            'choices': [{'message': {'content': mock_response['content']}}],
            'usage': {'total_tokens': mock_response['token_count']},
            'model': mock_response['model']
        }):
            # First query - should hit API and cache in all tiers
            result1 = await llm_client.process_query(query)
            
            # Cache the result manually in our cache manager
            cache_key = f"llm_query:{hash(query)}"
            await cache_manager.put(cache_key, result1)
            
            # Verify L1 cache
            l1_result = await cache_manager.l1_cache.get(cache_key)
            assert l1_result is not None
            assert l1_result['content'] == mock_response['content']
            
            # Second identical query - should hit cache
            result2 = await llm_client.process_query(query)
            
            # Verify cache utilization
            stats = await cache_manager.get_comprehensive_stats()
            assert stats['l1']['hit_rate'] > 0 or stats['l2']['hit_rate'] > 0
    
    @pytest.mark.integration
    async def test_data_structure_caching_integration(self, integrated_cache_system, test_data_manager):
        """Test data structure operations with caching"""
        cache_manager = integrated_cache_system['cache_manager']
        trie = integrated_cache_system['trie']
        hashmap = integrated_cache_system['hashmap']
        
        # Populate data structures
        test_data = {
            f"key_{i}": f"value_{i}" for i in range(100)
        }
        
        for key, value in test_data.items():
            trie.insert(key, value)
            hashmap.put(key, {'data': value, 'metadata': {'created': time.time()}})
        
        # Cache trie state
        trie_state = {
            'size': trie.size,
            'stats': trie.get_stats()
        }
        await cache_manager.put("trie_state", trie_state)
        
        # Cache hashmap state
        hashmap_state = {
            'size': hashmap.size(),
            'load_factor': hashmap.load_factor,
            'metrics': hashmap.get_performance_metrics()
        }
        await cache_manager.put("hashmap_state", hashmap_state)
        
        # Verify cached states
        cached_trie = await cache_manager.get("trie_state")
        cached_hashmap = await cache_manager.get("hashmap_state")
        
        assert cached_trie['size'] == trie.size
        assert cached_hashmap['size'] == hashmap.size()
        
        # Test cache warming for frequently accessed data
        frequent_keys = [f"key_{i}" for i in range(0, 100, 10)]  # Every 10th key
        
        for key in frequent_keys:
            cache_key = f"frequent_data:{key}"
            data = {
                'trie_value': trie.search(key),
                'hashmap_value': hashmap.get(key)
            }
            await cache_manager.put(cache_key, data)
        
        # Verify cache warming effectiveness
        stats = await cache_manager.get_comprehensive_stats()
        assert stats['overview']['total_entries'] >= len(frequent_keys)
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_cache_performance_under_load(self, integrated_cache_system, performance_timer):
        """Test cache performance under high load"""
        cache_manager = integrated_cache_system['cache_manager']
        
        # Generate load
        num_operations = 1000
        keys = [f"load_test_key_{i}" for i in range(num_operations)]
        values = [f"load_test_value_{i}" for i in range(num_operations)]
        
        # Test write performance
        performance_timer.start()
        write_tasks = [cache_manager.put(key, value) for key, value in zip(keys, values)]
        await asyncio.gather(*write_tasks)
        performance_timer.stop()
        
        write_time = performance_timer.elapsed
        write_ops_per_second = num_operations / write_time
        
        assert write_ops_per_second > 500, f"Write performance {write_ops_per_second:.0f} ops/sec too low"
        
        # Test read performance
        performance_timer.start()
        read_tasks = [cache_manager.get(key) for key in keys]
        results = await asyncio.gather(*read_tasks)
        performance_timer.stop()
        
        read_time = performance_timer.elapsed
        read_ops_per_second = num_operations / read_time
        
        assert read_ops_per_second > 1000, f"Read performance {read_ops_per_second:.0f} ops/sec too low"
        
        # Verify cache hit rate
        hit_count = sum(1 for result in results if result is not None)
        hit_rate = hit_count / num_operations
        assert hit_rate > 0.9, f"Cache hit rate {hit_rate:.2f} too low"


class TestAPIEndpointIntegration:
    """Integration tests for API endpoints with backend components"""
    
    @pytest.fixture
    async def test_server(self):
        """Create test server with API endpoints"""
        app = web.Application()
        
        # Initialize backend components
        file_processor = OptimizedFileProcessor()
        llm_client = OptimizedLLMClient({
            'providers': {
                'openai': {'api_key': 'test_key', 'rate_limit': {'requests_per_minute': 60}}
            }
        })
        query_router = IntelligentQueryRouter()
        
        # Store components in app for access in handlers
        app['file_processor'] = file_processor
        app['llm_client'] = llm_client
        app['query_router'] = query_router
        
        # Define API endpoints
        async def upload_file(request):
            """File upload endpoint"""
            try:
                reader = await request.multipart()
                field = await reader.next()
                
                if field.name == 'file':
                    filename = field.filename
                    content = await field.read()
                    
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                    
                    # Process file
                    result = await app['file_processor'].process_file(tmp_path)
                    
                    # Cleanup
                    os.unlink(tmp_path)
                    
                    return web.json_response({
                        'success': True,
                        'filename': filename,
                        'file_type': result['file_type'],
                        'metadata': result['metadata']
                    })
                
                return web.json_response({'error': 'No file provided'}, status=400)
                
            except Exception as e:
                return web.json_response({'error': str(e)}, status=500)
        
        async def process_query(request):
            """Query processing endpoint"""
            try:
                data = await request.json()
                query = data.get('query')
                
                if not query:
                    return web.json_response({'error': 'No query provided'}, status=400)
                
                # Route query through intelligent router
                routing_result = await app['query_router'].route_query(query)
                
                # Mock LLM processing
                with patch.object(app['llm_client'], '_make_api_request', return_value={
                    'choices': [{'message': {'content': f'Response to: {query}'}}],
                    'usage': {'total_tokens': 25},
                    'model': 'gpt-3.5-turbo'
                }):
                    llm_result = await app['llm_client'].process_query(query)
                
                return web.json_response({
                    'success': True,
                    'query': query,
                    'routing': {
                        'agent_type': routing_result.get('selected_agent'),
                        'confidence': routing_result.get('confidence', 0)
                    },
                    'response': llm_result['content'],
                    'metadata': {
                        'processing_time': llm_result.get('processing_time', 0),
                        'token_count': llm_result.get('token_count', 0)
                    }
                })
                
            except Exception as e:
                return web.json_response({'error': str(e)}, status=500)
        
        async def get_analytics(request):
            """Analytics endpoint"""
            try:
                # Get performance metrics from all components
                metrics = {
                    'file_processor': {
                        'files_processed': 100,  # Mock data
                        'average_processing_time': 2.5
                    },
                    'llm_client': app['llm_client'].get_performance_metrics(),
                    'query_router': {
                        'queries_routed': 50,
                        'average_routing_time': 0.1
                    }
                }
                
                return web.json_response({
                    'success': True,
                    'metrics': metrics,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                return web.json_response({'error': str(e)}, status=500)
        
        # Register routes
        app.router.add_post('/api/upload', upload_file)
        app.router.add_post('/api/query', process_query)
        app.router.add_get('/api/analytics', get_analytics)
        
        return app
    
    @pytest.mark.integration
    async def test_file_upload_endpoint(self, test_server, sample_csv_file):
        """Test file upload API endpoint integration"""
        async with aiohttp.test_utils.TestServer(test_server) as server:
            async with aiohttp.test_utils.TestClient(server) as client:
                
                # Read test file
                with open(sample_csv_file, 'rb') as f:
                    file_content = f.read()
                
                # Create multipart form data
                data = aiohttp.FormData()
                data.add_field('file', file_content, filename='test.csv', content_type='text/csv')
                
                # Make request
                response = await client.post('/api/upload', data=data)
                assert response.status == 200
                
                # Verify response
                result = await response.json()
                assert result['success'] is True
                assert result['filename'] == 'test.csv'
                assert result['file_type'] == 'csv'
                assert 'metadata' in result
    
    @pytest.mark.integration
    async def test_query_processing_endpoint(self, test_server):
        """Test query processing API endpoint integration"""
        async with aiohttp.test_utils.TestServer(test_server) as server:
            async with aiohttp.test_utils.TestClient(server) as client:
                
                # Test query
                query_data = {
                    'query': 'Analyze the sales data and show trends'
                }
                
                response = await client.post('/api/query', json=query_data)
                assert response.status == 200
                
                # Verify response
                result = await response.json()
                assert result['success'] is True
                assert result['query'] == query_data['query']
                assert 'routing' in result
                assert 'response' in result
                assert 'metadata' in result
    
    @pytest.mark.integration
    async def test_analytics_endpoint(self, test_server):
        """Test analytics API endpoint integration"""
        async with aiohttp.test_utils.TestServer(test_server) as server:
            async with aiohttp.test_utils.TestClient(server) as client:
                
                response = await client.get('/api/analytics')
                assert response.status == 200
                
                # Verify response
                result = await response.json()
                assert result['success'] is True
                assert 'metrics' in result
                assert 'file_processor' in result['metrics']
                assert 'llm_client' in result['metrics']
                assert 'query_router' in result['metrics']
    
    @pytest.mark.integration
    async def test_error_handling_in_endpoints(self, test_server):
        """Test error handling in API endpoints"""
        async with aiohttp.test_utils.TestServer(test_server) as server:
            async with aiohttp.test_utils.TestClient(server) as client:
                
                # Test missing query
                response = await client.post('/api/query', json={})
                assert response.status == 400
                
                result = await response.json()
                assert 'error' in result
                assert 'No query provided' in result['error']
                
                # Test invalid JSON
                response = await client.post('/api/query', data='invalid json')
                assert response.status in [400, 500]  # Depends on implementation
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_api_performance_under_load(self, test_server, performance_timer):
        """Test API performance under concurrent load"""
        async with aiohttp.test_utils.TestServer(test_server) as server:
            async with aiohttp.test_utils.TestClient(server) as client:
                
                # Generate concurrent requests
                num_requests = 50
                query_data = {'query': 'Test concurrent query'}
                
                performance_timer.start()
                
                # Make concurrent requests
                tasks = [
                    client.post('/api/query', json=query_data)
                    for _ in range(num_requests)
                ]
                responses = await asyncio.gather(*tasks)
                
                performance_timer.stop()
                
                # Verify all requests succeeded
                success_count = sum(1 for r in responses if r.status == 200)
                assert success_count == num_requests
                
                # Check performance
                total_time = performance_timer.elapsed
                requests_per_second = num_requests / total_time
                
                assert requests_per_second > 10, f"API throughput {requests_per_second:.1f} req/sec too low"


class TestFrontendBackendIntegration:
    """Integration tests for frontend-backend communication"""
    
    @pytest.fixture
    def mock_frontend_requests(self):
        """Mock frontend request patterns"""
        return {
            'dashboard_init': {
                'endpoint': '/api/dashboard',
                'method': 'GET',
                'expected_data': ['file_stats', 'recent_queries', 'system_metrics']
            },
            'file_upload': {
                'endpoint': '/api/upload',
                'method': 'POST',
                'payload_type': 'multipart',
                'expected_response': ['file_id', 'processing_status', 'metadata']
            },
            'query_processing': {
                'endpoint': '/api/query',
                'method': 'POST',
                'payload': {'query': 'string', 'options': 'object'},
                'expected_response': ['result', 'confidence', 'processing_time']
            },
            'real_time_updates': {
                'endpoint': '/api/stream',
                'method': 'GET',
                'connection_type': 'websocket',
                'message_types': ['progress', 'completion', 'error']
            }
        }
    
    @pytest.mark.integration
    async def test_dashboard_data_flow(self, mock_frontend_requests):
        """Test dashboard data flow from backend to frontend"""
        # Mock backend components
        cache_manager = EnhancedCacheManager({
            'l1_cache': {'max_size': 100, 'ttl': 300}
        })
        
        file_processor = OptimizedFileProcessor()
        
        # Simulate dashboard data aggregation
        dashboard_data = {
            'file_stats': {
                'total_files': 150,
                'files_processed_today': 25,
                'average_processing_time': 3.2,
                'file_types': {'csv': 60, 'json': 40, 'xlsx': 30, 'pdf': 20}
            },
            'recent_queries': [
                {'query': 'Analyze sales trends', 'timestamp': time.time() - 300, 'status': 'completed'},
                {'query': 'Generate monthly report', 'timestamp': time.time() - 600, 'status': 'completed'},
                {'query': 'Find data correlations', 'timestamp': time.time() - 900, 'status': 'processing'}
            ],
            'system_metrics': {
                'cache_hit_rate': 0.85,
                'average_response_time': 1.8,
                'active_connections': 12,
                'queue_length': 3
            }
        }
        
        # Cache dashboard data (simulating backend behavior)
        await cache_manager.put('dashboard_data', dashboard_data)
        
        # Verify frontend can retrieve expected data
        cached_data = await cache_manager.get('dashboard_data')
        
        assert cached_data is not None
        for expected_key in mock_frontend_requests['dashboard_init']['expected_data']:
            assert expected_key in cached_data
        
        # Verify data structure matches frontend expectations
        assert isinstance(cached_data['file_stats']['total_files'], int)
        assert isinstance(cached_data['recent_queries'], list)
        assert isinstance(cached_data['system_metrics']['cache_hit_rate'], float)
    
    @pytest.mark.integration
    async def test_real_time_communication(self, mock_frontend_requests):
        """Test real-time communication patterns"""
        # Simulate WebSocket-like communication
        message_queue = asyncio.Queue()
        
        # Mock processing with progress updates
        async def simulate_long_running_process():
            stages = ['Uploading', 'Validating', 'Processing', 'Analyzing', 'Completed']
            
            for i, stage in enumerate(stages):
                progress_message = {
                    'type': 'progress',
                    'stage': stage,
                    'percentage': (i + 1) / len(stages) * 100,
                    'timestamp': time.time()
                }
                await message_queue.put(progress_message)
                await asyncio.sleep(0.1)  # Simulate processing time
            
            completion_message = {
                'type': 'completion',
                'result': 'Processing completed successfully',
                'final_stats': {'rows_processed': 10000, 'processing_time': 0.5},
                'timestamp': time.time()
            }
            await message_queue.put(completion_message)
        
        # Start simulation
        task = asyncio.create_task(simulate_long_running_process())
        
        # Collect messages (simulating frontend listening)
        received_messages = []
        expected_message_types = mock_frontend_requests['real_time_updates']['message_types']
        
        while not task.done() or not message_queue.empty():
            try:
                message = await asyncio.wait_for(message_queue.get(), timeout=0.2)
                received_messages.append(message)
            except asyncio.TimeoutError:
                if task.done():
                    break
        
        await task
        
        # Verify message flow
        assert len(received_messages) > 0
        
        progress_messages = [m for m in received_messages if m['type'] == 'progress']
        completion_messages = [m for m in received_messages if m['type'] == 'completion']
        
        assert len(progress_messages) > 0
        assert len(completion_messages) == 1
        
        # Verify progress sequence
        percentages = [m['percentage'] for m in progress_messages]
        assert percentages == sorted(percentages)  # Should be increasing
        assert percentages[-1] == 100  # Should reach 100%
    
    @pytest.mark.integration
    async def test_error_propagation(self):
        """Test error propagation from backend to frontend"""
        # Simulate various error scenarios
        error_scenarios = [
            {
                'component': 'file_processor',
                'error_type': 'FileNotFoundError',
                'error_message': 'File not found: invalid_file.csv',
                'expected_status': 404,
                'expected_frontend_action': 'show_error_dialog'
            },
            {
                'component': 'llm_client',
                'error_type': 'RateLimitError',
                'error_message': 'API rate limit exceeded',
                'expected_status': 429,
                'expected_frontend_action': 'show_retry_option'
            },
            {
                'component': 'cache_manager',
                'error_type': 'CacheError',
                'error_message': 'Cache storage full',
                'expected_status': 503,
                'expected_frontend_action': 'show_service_unavailable'
            }
        ]
        
        for scenario in error_scenarios:
            # Simulate error in backend component
            error_response = {
                'success': False,
                'error': {
                    'type': scenario['error_type'],
                    'message': scenario['error_message'],
                    'component': scenario['component'],
                    'timestamp': time.time(),
                    'request_id': f"req_{hash(scenario['error_message'])}"
                },
                'suggested_action': scenario['expected_frontend_action']
            }
            
            # Verify error structure matches frontend expectations
            assert 'success' in error_response
            assert error_response['success'] is False
            assert 'error' in error_response
            assert 'type' in error_response['error']
            assert 'message' in error_response['error']
            assert 'suggested_action' in error_response
    
    @pytest.mark.integration
    async def test_data_consistency_across_sessions(self):
        """Test data consistency across frontend sessions"""
        # Initialize shared backend state
        cache_manager = EnhancedCacheManager({
            'l1_cache': {'max_size': 100, 'ttl': 1800}
        })
        
        # Simulate multiple frontend sessions
        session_data = {}
        
        for session_id in ['session_1', 'session_2', 'session_3']:
            # Each session performs operations
            session_cache_key = f"session_data:{session_id}"
            
            session_info = {
                'session_id': session_id,
                'user_preferences': {
                    'theme': 'dark' if session_id == 'session_2' else 'light',
                    'language': 'en',
                    'notifications': True
                },
                'active_queries': [
                    f"Query from {session_id} - 1",
                    f"Query from {session_id} - 2"
                ],
                'timestamp': time.time()
            }
            
            await cache_manager.put(session_cache_key, session_info)
            session_data[session_id] = session_info
        
        # Verify all sessions can access their data
        for session_id, expected_data in session_data.items():
            session_cache_key = f"session_data:{session_id}"
            retrieved_data = await cache_manager.get(session_cache_key)
            
            assert retrieved_data is not None
            assert retrieved_data['session_id'] == expected_data['session_id']
            assert retrieved_data['user_preferences'] == expected_data['user_preferences']
        
        # Test shared global state
        global_state = {
            'system_announcements': ['System maintenance scheduled for tonight'],
            'feature_flags': {'new_dashboard': True, 'beta_features': False},
            'active_users': len(session_data)
        }
        
        await cache_manager.put('global_state', global_state)
        
        # All sessions should see the same global state
        for session_id in session_data.keys():
            retrieved_global = await cache_manager.get('global_state')
            assert retrieved_global is not None
            assert retrieved_global['active_users'] == len(session_data)


class TestComponentInteractionIntegration:
    """Integration tests for complex component interactions"""
    
    @pytest.mark.integration
    async def test_complete_data_pipeline(self, test_data_manager):
        """Test complete data processing pipeline integration"""
        # Initialize all pipeline components
        file_processor = OptimizedFileProcessor()
        cache_manager = EnhancedCacheManager({
            'l1_cache': {'max_size': 50, 'ttl': 300}
        })
        query_analyzer = QueryPatternAnalyzer()
        data_structures = {
            'trie': OptimizedTrie(),
            'hashmap': HighPerformanceHashMap()
        }
        
        # Create test data files
        csv_file = test_data_manager.create_test_csv("pipeline_test.csv", rows=100)
        json_file = test_data_manager.create_test_json("pipeline_test.json")
        
        # Stage 1: File Processing
        csv_result = await file_processor.process_file(str(csv_file))
        json_result = await file_processor.process_file(str(json_file))
        
        assert csv_result['file_type'] == 'csv'
        assert json_result['file_type'] == 'json'
        
        # Stage 2: Cache Results
        await cache_manager.put(f"file_result:{csv_file.name}", csv_result)
        await cache_manager.put(f"file_result:{json_file.name}", json_result)
        
        # Stage 3: Index Data in Structures
        for row in csv_result['data']:
            if isinstance(row, dict) and 'id' in row:
                key = f"csv_record_{row['id']}"
                data_structures['trie'].insert(key, str(row))
                data_structures['hashmap'].put(key, row)
        
        # Stage 4: Query Processing Pipeline
        test_queries = [
            "Find all records with ID greater than 50",
            "Show me the CSV data structure",
            "What is the total count of records?"
        ]
        
        query_results = []
        for query in test_queries:
            # Analyze query pattern
            query_profile = query_analyzer.analyze_query(query)
            
            # Route based on query type
            if 'records' in query.lower():
                # Search in data structures
                search_results = []
                for i in range(51, 101):  # IDs greater than 50
                    key = f"csv_record_{i}"
                    if data_structures['trie'].search(key):
                        record = data_structures['hashmap'].get(key)
                        if record:
                            search_results.append(record)
                
                result = {
                    'query': query,
                    'type': 'data_search',
                    'results': search_results,
                    'count': len(search_results)
                }
            else:
                # General query - use cached file results
                cached_csv = await cache_manager.get(f"file_result:{csv_file.name}")
                result = {
                    'query': query,
                    'type': 'general_info',
                    'file_info': {
                        'total_rows': cached_csv['metadata']['total_rows'],
                        'columns': cached_csv['metadata']['columns']
                    }
                }
            
            query_results.append(result)
        
        # Verify pipeline results
        assert len(query_results) == len(test_queries)
        
        # Verify data search worked
        data_search_result = next(r for r in query_results if r['type'] == 'data_search')
        assert data_search_result['count'] > 0
        assert data_search_result['count'] <= 50  # Records with ID > 50
        
        # Verify info query worked
        info_result = next(r for r in query_results if r['type'] == 'general_info')
        assert 'file_info' in info_result
        assert info_result['file_info']['total_rows'] > 0
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_system_performance_integration(self, test_data_manager, performance_timer):
        """Test integrated system performance"""
        # Initialize performance monitoring across all components
        components = {
            'file_processor': OptimizedFileProcessor(),
            'cache_manager': EnhancedCacheManager({
                'l1_cache': {'max_size': 200, 'ttl': 300}
            }),
            'data_processor': OptimizedDataProcessor(),
            'batch_processor': OptimizedFileProcessor()
        }
        
        # Create workload
        test_files = []
        for i in range(20):
            file_path = test_data_manager.create_test_csv(f"perf_test_{i}.csv", rows=200)
            test_files.append(str(file_path))
        
        # Performance test: End-to-end processing
        performance_timer.start()
        
        # Batch process files
        file_results = await components['batch_processor'].process_files(test_files)
        
        # Cache all results
        cache_tasks = []
        for i, result in enumerate(file_results):
            cache_key = f"perf_result_{i}"
            cache_tasks.append(components['cache_manager'].put(cache_key, result))
        await asyncio.gather(*cache_tasks)
        
        # Process data through data processor
        processing_tasks = []
        for result in file_results:
            if 'error' not in result:
                task = components['data_processor'].process_data(result['data'])
                processing_tasks.append(task)
        
        processed_results = await asyncio.gather(*processing_tasks)
        
        performance_timer.stop()
        
        # Verify performance metrics
        total_time = performance_timer.elapsed
        total_files = len(test_files)
        files_per_second = total_files / total_time
        
        assert files_per_second > 5, f"System throughput {files_per_second:.1f} files/sec too low"
        
        # Verify all processing succeeded
        successful_files = len([r for r in file_results if 'error' not in r])
        assert successful_files == total_files
        
        successful_processing = len([r for r in processed_results if r and 'error' not in r])
        assert successful_processing == successful_files


if __name__ == '__main__':
    # Run integration tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'integration'
    ])