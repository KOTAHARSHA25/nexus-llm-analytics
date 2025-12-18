# Performance Tests
# Production-grade performance testing for benchmarking optimization gains

import pytest
import asyncio
import time
import statistics
import psutil
import os
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
import sys

# Import optimized components for performance testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_data_structures import OptimizedTrie, HighPerformanceHashMap, OptimizedDataProcessor
from backend.core.optimized_llm_client import OptimizedLLMClient, SmartCache
from backend.core.optimized_file_io import OptimizedFileProcessor, StreamingCSVReader
from backend.core.enhanced_cache_integration import EnhancedCacheManager
from backend.core.intelligent_query_engine import IntelligentQueryRouter

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation: str
    duration: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    percentile_95: float = 0.0
    percentile_99: float = 0.0

class PerformanceBenchmark:
    """Performance benchmarking utility"""
    
    def __init__(self):
        self.results = []
        self.baseline_metrics = {}
    
    def measure_performance(self, operation_name: str, operation_func, *args, **kwargs):
        """Measure performance of an operation"""
        process = psutil.Process(os.getpid())
        
        # Initial measurements
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        start_time = time.perf_counter()
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
            error_count = 0
        except Exception as e:
            result = None
            success = False
            error_count = 1
        
        end_time = time.perf_counter()
        
        # Final measurements
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent()
        
        duration = end_time - start_time
        memory_usage = final_memory - initial_memory
        cpu_usage = (initial_cpu + final_cpu) / 2
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration=duration,
            throughput=1.0 / duration if duration > 0 else 0,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            success_rate=1.0 if success else 0.0,
            error_count=error_count
        )
        
        self.results.append(metrics)
        return result, metrics
    
    async def measure_async_performance(self, operation_name: str, operation_func, *args, **kwargs):
        """Measure performance of an async operation"""
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss / 1024 / 1024
        initial_cpu = process.cpu_percent()
        
        start_time = time.perf_counter()
        
        try:
            result = await operation_func(*args, **kwargs)
            success = True
            error_count = 0
        except Exception as e:
            result = None
            success = False
            error_count = 1
        
        end_time = time.perf_counter()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        final_cpu = process.cpu_percent()
        
        duration = end_time - start_time
        memory_usage = final_memory - initial_memory
        cpu_usage = (initial_cpu + final_cpu) / 2
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            duration=duration,
            throughput=1.0 / duration if duration > 0 else 0,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            success_rate=1.0 if success else 0.0,
            error_count=error_count
        )
        
        self.results.append(metrics)
        return result, metrics
    
    def benchmark_multiple_runs(self, operation_name: str, operation_func, runs: int = 10, *args, **kwargs):
        """Benchmark operation over multiple runs"""
        durations = []
        throughputs = []
        memory_usages = []
        error_count = 0
        
        for i in range(runs):
            try:
                result, metrics = self.measure_performance(f"{operation_name}_run_{i}", operation_func, *args, **kwargs)
                durations.append(metrics.duration)
                throughputs.append(metrics.throughput)
                memory_usages.append(metrics.memory_usage_mb)
                error_count += metrics.error_count
            except Exception:
                error_count += 1
        
        if durations:
            aggregate_metrics = PerformanceMetrics(
                operation=operation_name,
                duration=statistics.mean(durations),
                throughput=statistics.mean(throughputs),
                memory_usage_mb=statistics.mean(memory_usages),
                cpu_usage_percent=0,  # Not meaningful for aggregate
                success_rate=(runs - error_count) / runs,
                error_count=error_count,
                percentile_95=np.percentile(durations, 95),
                percentile_99=np.percentile(durations, 99)
            )
            
            self.results.append(aggregate_metrics)
            return aggregate_metrics
        
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            return {"error": "No performance data available"}
        
        report = {
            "summary": {
                "total_operations": len(self.results),
                "average_duration": statistics.mean([r.duration for r in self.results]),
                "average_throughput": statistics.mean([r.throughput for r in self.results]),
                "total_memory_usage": sum([r.memory_usage_mb for r in self.results]),
                "overall_success_rate": statistics.mean([r.success_rate for r in self.results])
            },
            "operations": []
        }
        
        for result in self.results:
            report["operations"].append({
                "operation": result.operation,
                "duration_ms": result.duration * 1000,
                "throughput_ops_per_sec": result.throughput,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
                "success_rate": result.success_rate,
                "error_count": result.error_count,
                "percentile_95_ms": result.percentile_95 * 1000,
                "percentile_99_ms": result.percentile_99 * 1000
            })
        
        return report


class TestDataStructurePerformance:
    """Performance tests for optimized data structures"""
    
    @pytest.fixture
    def benchmark(self):
        """Create performance benchmark instance"""
        return PerformanceBenchmark()
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing"""
        return {f"key_{i}": f"value_{i}" for i in range(100000)}
    
    @pytest.mark.performance
    def test_trie_insertion_performance(self, benchmark, large_dataset):
        """Test trie insertion performance"""
        trie = OptimizedTrie()
        
        def insert_batch():
            for key, value in large_dataset.items():
                trie.insert(key, value)
            return trie.size
        
        result, metrics = benchmark.measure_performance("trie_bulk_insertion", insert_batch)
        
        # Performance assertions
        assert metrics.duration < 10.0, f"Trie insertion took {metrics.duration:.2f}s, expected < 10s"
        assert metrics.throughput > 10000, f"Trie insertion throughput {metrics.throughput:.0f} ops/sec too low"
        assert result == len(large_dataset)
    
    @pytest.mark.performance
    def test_trie_search_performance(self, benchmark, large_dataset):
        """Test trie search performance"""
        trie = OptimizedTrie()
        
        # Populate trie
        for key, value in large_dataset.items():
            trie.insert(key, value)
        
        # Test search performance
        search_keys = list(large_dataset.keys())[::10]  # Every 10th key
        
        def search_batch():
            found_count = 0
            for key in search_keys:
                if trie.search(key):
                    found_count += 1
            return found_count
        
        result, metrics = benchmark.measure_performance("trie_batch_search", search_batch)
        
        # Performance assertions
        assert metrics.throughput > 50000, f"Trie search throughput {metrics.throughput:.0f} ops/sec too low"
        assert result == len(search_keys)  # All keys should be found
    
    @pytest.mark.performance
    def test_hashmap_performance_comparison(self, benchmark):
        """Test hashmap performance against standard dict"""
        test_size = 50000
        test_data = {f"perf_key_{i}": f"perf_value_{i}" for i in range(test_size)}
        
        # Test optimized hashmap
        optimized_hm = HighPerformanceHashMap()
        
        def optimized_operations():
            # Insertion
            for key, value in test_data.items():
                optimized_hm.put(key, value)
            
            # Lookups
            found_count = 0
            for key in list(test_data.keys())[::5]:  # Every 5th key
                if optimized_hm.get(key):
                    found_count += 1
            
            return found_count
        
        opt_result, opt_metrics = benchmark.measure_performance("optimized_hashmap", optimized_operations)
        
        # Test standard dict
        standard_dict = {}
        
        def standard_operations():
            # Insertion
            for key, value in test_data.items():
                standard_dict[key] = value
            
            # Lookups
            found_count = 0
            for key in list(test_data.keys())[::5]:
                if key in standard_dict:
                    found_count += 1
            
            return found_count
        
        std_result, std_metrics = benchmark.measure_performance("standard_dict", standard_operations)
        
        # Compare performance
        speedup = std_metrics.duration / opt_metrics.duration
        print(f"Optimized hashmap speedup: {speedup:.2f}x")
        
        # Should be competitive or better
        assert speedup >= 0.8, f"Optimized hashmap is {1/speedup:.2f}x slower than standard dict"
    
    @pytest.mark.performance
    def test_data_processor_parallel_performance(self, benchmark, test_data_manager):
        """Test data processor parallel performance"""
        processor = OptimizedDataProcessor()
        
        # Create large dataset
        large_data = []
        for i in range(20000):
            large_data.append({
                'id': i,
                'value': i * 1.5,
                'category': f'cat_{i % 10}',
                'description': f'Description for item {i} with additional text'
            })
        
        # Test sequential processing
        def sequential_processing():
            return processor.process_data(large_data)
        
        seq_result, seq_metrics = benchmark.measure_performance("sequential_processing", sequential_processing)
        
        # Test parallel processing
        def parallel_processing():
            return processor.process_data_parallel(large_data, num_workers=4)
        
        par_result, par_metrics = benchmark.measure_performance("parallel_processing", parallel_processing)
        
        # Compare performance
        if seq_metrics.duration > 1.0:  # Only compare if sequential takes significant time
            speedup = seq_metrics.duration / par_metrics.duration
            print(f"Parallel processing speedup: {speedup:.2f}x")
            assert speedup > 1.2, f"Parallel processing speedup {speedup:.2f}x insufficient"
    
    @pytest.mark.performance
    def test_memory_efficiency_under_load(self, benchmark):
        """Test memory efficiency under high load"""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Create multiple data structures
        structures = {
            'tries': [OptimizedTrie() for _ in range(10)],
            'hashmaps': [HighPerformanceHashMap() for _ in range(10)]
        }
        
        def memory_load_test():
            # Populate all structures
            for i in range(1000):
                key = f"memory_test_key_{i}"
                value = f"memory_test_value_{i}_with_extra_data_to_increase_size"
                
                for trie in structures['tries']:
                    trie.insert(key, value)
                
                for hm in structures['hashmaps']:
                    hm.put(key, value)
            
            # Perform operations
            total_operations = 0
            for trie in structures['tries']:
                total_operations += trie.size
            
            for hm in structures['hashmaps']:
                total_operations += hm.size()
            
            return total_operations
        
        result, metrics = benchmark.measure_performance("memory_load_test", memory_load_test)
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory efficiency checks
        memory_per_operation = memory_increase / result if result > 0 else 0
        assert memory_per_operation < 0.001, f"Memory per operation {memory_per_operation:.6f}MB too high"


class TestFileIOPerformance:
    """Performance tests for file I/O operations"""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    @pytest.mark.performance
    async def test_csv_streaming_performance(self, benchmark, test_data_manager):
        """Test CSV streaming performance"""
        # Create large CSV file
        large_csv = test_data_manager.create_large_test_file("streaming_perf.csv", size_mb=100)
        
        csv_reader = StreamingCSVReader(chunk_size=5000)
        
        async def streaming_read():
            total_rows = 0
            chunk_count = 0
            
            async for chunk in csv_reader.stream_chunks(str(large_csv)):
                total_rows += len(chunk['data'])
                chunk_count += 1
            
            return total_rows
        
        result, metrics = await benchmark.measure_async_performance("csv_streaming", streaming_read)
        
        # Performance assertions
        file_size_mb = os.path.getsize(large_csv) / 1024 / 1024
        throughput_mb_per_sec = file_size_mb / metrics.duration
        
        assert throughput_mb_per_sec > 10, f"CSV streaming throughput {throughput_mb_per_sec:.1f} MB/s too low"
        assert result > 0, "Should have read some rows"
    
    @pytest.mark.performance
    async def test_batch_file_processing_performance(self, benchmark, test_data_manager):
        """Test batch file processing performance"""
        # Create multiple test files
        test_files = []
        for i in range(50):
            file_path = test_data_manager.create_test_csv(f"batch_perf_{i}.csv", rows=500)
            test_files.append(str(file_path))
        
        batch_processor = OptimizedFileProcessor()
        
        async def batch_processing():
            results = await batch_processor.process_files(test_files)
            successful = [r for r in results if 'error' not in r]
            return len(successful)
        
        result, metrics = await benchmark.measure_async_performance("batch_processing", batch_processing)
        
        # Performance assertions
        files_per_second = len(test_files) / metrics.duration
        assert files_per_second > 5, f"Batch processing {files_per_second:.1f} files/sec too low"
        assert result == len(test_files), "All files should be processed successfully"
    
    @pytest.mark.performance
    async def test_file_type_detection_performance(self, benchmark, test_data_manager):
        """Test file type detection performance"""
        from backend.core.optimized_file_io import FileTypeDetector
        
        detector = FileTypeDetector()
        
        # Create files of different types
        test_files = []
        test_files.append(test_data_manager.create_test_csv("detect_perf.csv", rows=100))
        test_files.append(test_data_manager.create_test_json("detect_perf.json"))
        test_files.append(test_data_manager.create_test_jsonl("detect_perf.jsonl", records=50))
        
        # Multiply files for performance testing
        all_test_files = test_files * 100  # 300 files total
        
        def detection_batch():
            detected_types = []
            for file_path in all_test_files:
                result = detector.detect_file_type(str(file_path))
                detected_types.append(result['file_type'])
            return len(detected_types)
        
        result, metrics = benchmark.measure_performance("file_type_detection", detection_batch)
        
        # Performance assertions
        detections_per_second = result / metrics.duration
        assert detections_per_second > 100, f"Detection rate {detections_per_second:.0f}/sec too low"


class TestLLMClientPerformance:
    """Performance tests for LLM client operations"""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    @pytest.fixture
    def llm_client(self):
        """Create LLM client for performance testing"""
        config = {
            'providers': {
                'openai': {
                    'api_key': 'test_key',
                    'rate_limit': {'requests_per_minute': 1000}
                }
            },
            'cache': {'max_size': 1000, 'ttl': 3600},
            'connection_pool': {'max_connections': 20}
        }
        return OptimizedLLMClient(config)
    
    @pytest.mark.performance
    async def test_concurrent_query_performance(self, benchmark, llm_client):
        """Test concurrent query processing performance"""
        # Mock LLM responses
        mock_response = {
            'choices': [{'message': {'content': 'Performance test response'}}],
            'usage': {'total_tokens': 30},
            'model': 'gpt-3.5-turbo'
        }
        
        queries = [f"Performance test query {i}" for i in range(100)]
        
        async def concurrent_processing():
            with patch.object(llm_client, '_make_api_request', return_value=mock_response):
                # Process queries concurrently
                tasks = [llm_client.process_query(query) for query in queries]
                results = await asyncio.gather(*tasks)
                return len([r for r in results if 'content' in r])
        
        result, metrics = await benchmark.measure_async_performance("concurrent_queries", concurrent_processing)
        
        # Performance assertions
        queries_per_second = len(queries) / metrics.duration
        assert queries_per_second > 50, f"Concurrent processing {queries_per_second:.1f} queries/sec too low"
        assert result == len(queries), "All queries should be processed"
    
    @pytest.mark.performance
    async def test_cache_performance_impact(self, benchmark, llm_client):
        """Test cache performance impact"""
        mock_response = {
            'choices': [{'message': {'content': 'Cached response test'}}],
            'usage': {'total_tokens': 25},
            'model': 'gpt-3.5-turbo'
        }
        
        # Test without cache (first run)
        query = "Cache performance test query"
        
        async def first_query():
            with patch.object(llm_client, '_make_api_request', return_value=mock_response):
                result = await llm_client.process_query(query)
                return result
        
        result1, metrics1 = await benchmark.measure_async_performance("first_query", first_query)
        
        # Test with cache (second run - same query)
        async def cached_query():
            result = await llm_client.process_query(query)
            return result
        
        result2, metrics2 = await benchmark.measure_async_performance("cached_query", cached_query)
        
        # Cache should significantly improve performance
        if metrics1.duration > 0.01:  # Only compare if first query took meaningful time
            speedup = metrics1.duration / metrics2.duration
            print(f"Cache speedup: {speedup:.2f}x")
            assert speedup > 5, f"Cache speedup {speedup:.2f}x insufficient"
    
    @pytest.mark.performance
    async def test_batch_processing_performance(self, benchmark, llm_client):
        """Test batch processing performance"""
        mock_response = {
            'choices': [{'message': {'content': 'Batch response'}}],
            'usage': {'total_tokens': 20},
            'model': 'gpt-3.5-turbo'
        }
        
        batch_queries = [f"Batch query {i}" for i in range(50)]
        
        async def batch_processing():
            with patch.object(llm_client, '_make_api_request', return_value=mock_response):
                results = await llm_client.process_batch(batch_queries)
                return len(results)
        
        result, metrics = await benchmark.measure_async_performance("batch_processing", batch_processing)
        
        # Performance assertions
        batch_throughput = len(batch_queries) / metrics.duration
        assert batch_throughput > 25, f"Batch throughput {batch_throughput:.1f} queries/sec too low"
        assert result == len(batch_queries)


class TestCachePerformance:
    """Performance tests for caching system"""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager for performance testing"""
        config = {
            'l1_cache': {'max_size': 1000, 'ttl': 300},
            'l2_cache': {'max_size': 5000, 'ttl': 1800},
            'l3_cache': {'max_size': 10000, 'ttl': 3600}
        }
        return EnhancedCacheManager(config)
    
    @pytest.mark.performance
    async def test_multi_tier_cache_performance(self, benchmark, cache_manager):
        """Test multi-tier cache performance"""
        # Prepare test data
        test_data = {f"cache_key_{i}": f"cache_value_{i}_with_substantial_content" for i in range(10000)}
        
        async def cache_write_performance():
            write_count = 0
            for key, value in test_data.items():
                await cache_manager.put(key, value)
                write_count += 1
            return write_count
        
        write_result, write_metrics = await benchmark.measure_async_performance("cache_writes", cache_write_performance)
        
        # Test read performance
        read_keys = list(test_data.keys())[::5]  # Every 5th key
        
        async def cache_read_performance():
            read_count = 0
            for key in read_keys:
                value = await cache_manager.get(key)
                if value is not None:
                    read_count += 1
            return read_count
        
        read_result, read_metrics = await benchmark.measure_async_performance("cache_reads", cache_read_performance)
        
        # Performance assertions
        write_ops_per_sec = write_result / write_metrics.duration
        read_ops_per_sec = read_result / read_metrics.duration
        
        assert write_ops_per_sec > 1000, f"Cache write performance {write_ops_per_sec:.0f} ops/sec too low"
        assert read_ops_per_sec > 5000, f"Cache read performance {read_ops_per_sec:.0f} ops/sec too low"
    
    @pytest.mark.performance
    async def test_cache_hit_rate_performance(self, benchmark, cache_manager):
        """Test cache hit rate under load"""
        # Populate cache
        initial_data = {f"hit_test_key_{i}": f"hit_test_value_{i}" for i in range(1000)}
        
        for key, value in initial_data.items():
            await cache_manager.put(key, value)
        
        # Test mixed read pattern (some hits, some misses)
        read_keys = []
        for i in range(2000):
            if i < 800:  # 80% should be hits
                read_keys.append(f"hit_test_key_{i % 1000}")
            else:  # 20% should be misses
                read_keys.append(f"miss_key_{i}")
        
        async def mixed_read_pattern():
            hit_count = 0
            total_reads = 0
            
            for key in read_keys:
                value = await cache_manager.get(key)
                if value is not None:
                    hit_count += 1
                total_reads += 1
            
            return hit_count, total_reads
        
        (hits, total), metrics = await benchmark.measure_async_performance("mixed_reads", mixed_read_pattern)
        
        # Performance and accuracy assertions
        hit_rate = hits / total
        reads_per_sec = total / metrics.duration
        
        assert hit_rate > 0.75, f"Cache hit rate {hit_rate:.2f} too low"
        assert reads_per_sec > 2000, f"Read performance {reads_per_sec:.0f} ops/sec too low"


class TestSystemScalabilityPerformance:
    """Performance tests for system scalability"""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_load_scalability(self, benchmark, test_data_manager):
        """Test system performance under increasing load"""
        # Initialize system components
        cache_manager = EnhancedCacheManager({
            'l1_cache': {'max_size': 500, 'ttl': 300}
        })
        file_processor = OptimizedFileProcessor()
        query_router = IntelligentQueryRouter()
        
        # Test different load levels
        load_levels = [10, 50, 100, 200]
        scalability_results = []
        
        for load_level in load_levels:
            # Create test files for this load level
            test_files = []
            for i in range(load_level):
                file_path = test_data_manager.create_test_csv(f"scalability_{load_level}_{i}.csv", rows=100)
                test_files.append(str(file_path))
            
            async def load_test():
                # Process files
                file_tasks = [file_processor.process_file(file_path) for file_path in test_files]
                file_results = await asyncio.gather(*file_tasks)
                
                # Cache results
                cache_tasks = []
                for i, result in enumerate(file_results):
                    cache_key = f"load_test_{load_level}_{i}"
                    cache_tasks.append(cache_manager.put(cache_key, result))
                await asyncio.gather(*cache_tasks)
                
                # Process queries
                queries = [f"Analyze data from load test {load_level} file {i}" for i in range(min(load_level, 20))]
                query_tasks = []
                for query in queries:
                    task = query_router.route_query(query)
                    query_tasks.append(task)
                
                query_results = await asyncio.gather(*query_tasks)
                
                return len(file_results), len(query_results)
            
            (files_processed, queries_processed), metrics = await benchmark.measure_async_performance(
                f"load_test_{load_level}", load_test
            )
            
            scalability_results.append({
                'load_level': load_level,
                'files_processed': files_processed,
                'queries_processed': queries_processed,
                'duration': metrics.duration,
                'throughput': files_processed / metrics.duration,
                'memory_usage': metrics.memory_usage_mb
            })
        
        # Analyze scalability
        print("\nScalability Results:")
        for result in scalability_results:
            print(f"Load {result['load_level']}: {result['throughput']:.1f} files/sec, "
                  f"{result['memory_usage']:.1f}MB memory")
        
        # Check that system scales reasonably
        throughputs = [r['throughput'] for r in scalability_results]
        
        # Throughput should not degrade drastically with increased load
        for i in range(1, len(throughputs)):
            degradation = (throughputs[0] - throughputs[i]) / throughputs[0]
            assert degradation < 0.75, f"Throughput degraded by {degradation*100:.1f}% at load level {load_levels[i]}"
    
    @pytest.mark.performance
    def test_concurrent_user_simulation(self, benchmark):
        """Test performance under concurrent user simulation"""
        # Simulate concurrent users
        num_users = 20
        operations_per_user = 10
        
        def simulate_user_operations(user_id):
            """Simulate operations for a single user"""
            operations_completed = 0
            
            # Initialize user-specific components
            user_cache = SmartCache(max_size=50, ttl=300)
            user_trie = OptimizedTrie()
            
            try:
                for op in range(operations_per_user):
                    # Simulate different operations
                    if op % 3 == 0:
                        # Cache operation
                        user_cache.put(f"user_{user_id}_op_{op}", f"data_{op}")
                        result = user_cache.get(f"user_{user_id}_op_{op}")
                    elif op % 3 == 1:
                        # Trie operation
                        user_trie.insert(f"user_{user_id}_key_{op}", f"value_{op}")
                        result = user_trie.search(f"user_{user_id}_key_{op}")
                    else:
                        # Mixed operation
                        key = f"mixed_{user_id}_{op}"
                        user_trie.insert(key, f"trie_value_{op}")
                        user_cache.put(key, f"cache_value_{op}")
                    
                    operations_completed += 1
                    time.sleep(0.01)  # Simulate processing time
                
                return operations_completed
                
            except Exception as e:
                return operations_completed  # Return partial completion
        
        def concurrent_user_test():
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(simulate_user_operations, user_id) for user_id in range(num_users)]
                results = [future.result() for future in as_completed(futures)]
            
            return sum(results)
        
        result, metrics = benchmark.measure_performance("concurrent_users", concurrent_user_test)
        
        # Performance assertions
        total_expected_operations = num_users * operations_per_user
        success_rate = result / total_expected_operations
        operations_per_second = result / metrics.duration
        
        assert success_rate > 0.9, f"Success rate {success_rate:.2f} too low under concurrent load"
        assert operations_per_second > 100, f"Concurrent throughput {operations_per_second:.0f} ops/sec too low"


class TestRegressionPerformance:
    """Performance regression tests"""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    @pytest.mark.performance
    def test_performance_regression_baseline(self, benchmark):
        """Establish performance baseline for regression testing"""
        # Define baseline operations
        baseline_operations = {
            'trie_operations': self._trie_baseline_test,
            'hashmap_operations': self._hashmap_baseline_test,
            'cache_operations': self._cache_baseline_test
        }
        
        baseline_results = {}
        
        for operation_name, operation_func in baseline_operations.items():
            # Run multiple times for stable baseline
            metrics = benchmark.benchmark_multiple_runs(operation_name, operation_func, runs=5)
            
            if metrics:
                baseline_results[operation_name] = {
                    'duration': metrics.duration,
                    'throughput': metrics.throughput,
                    'memory_usage': metrics.memory_usage_mb,
                    'p95_duration': metrics.percentile_95,
                    'p99_duration': metrics.percentile_99
                }
        
        # Save baseline for comparison (in real scenario, save to file)
        benchmark.baseline_metrics = baseline_results
        
        # Verify baseline is reasonable
        for operation, metrics in baseline_results.items():
            assert metrics['throughput'] > 0, f"Baseline throughput for {operation} is invalid"
            assert metrics['duration'] > 0, f"Baseline duration for {operation} is invalid"
        
        print("\nPerformance Baseline Established:")
        for operation, metrics in baseline_results.items():
            print(f"{operation}: {metrics['throughput']:.0f} ops/sec, "
                  f"{metrics['duration']*1000:.1f}ms avg, "
                  f"P95: {metrics['p95_duration']*1000:.1f}ms")
    
    def _trie_baseline_test(self):
        """Baseline test for trie operations"""
        trie = OptimizedTrie()
        
        # Insert operations
        for i in range(1000):
            trie.insert(f"baseline_key_{i}", f"baseline_value_{i}")
        
        # Search operations
        found_count = 0
        for i in range(0, 1000, 5):
            if trie.search(f"baseline_key_{i}"):
                found_count += 1
        
        return found_count
    
    def _hashmap_baseline_test(self):
        """Baseline test for hashmap operations"""
        hashmap = HighPerformanceHashMap()
        
        # Insert operations
        for i in range(1000):
            hashmap.put(f"baseline_key_{i}", f"baseline_value_{i}")
        
        # Lookup operations
        found_count = 0
        for i in range(0, 1000, 5):
            if hashmap.get(f"baseline_key_{i}"):
                found_count += 1
        
        return found_count
    
    async def _cache_baseline_test(self):
        """Baseline test for cache operations"""
        cache = SmartCache(max_size=500, ttl=3600)
        
        # Write operations
        for i in range(500):
            cache.put(f"baseline_cache_key_{i}", f"baseline_cache_value_{i}")
        
        # Read operations
        hit_count = 0
        for i in range(0, 500, 3):
            if cache.get(f"baseline_cache_key_{i}"):
                hit_count += 1
        
        return hit_count


if __name__ == '__main__':
    # Run performance tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'performance',
        '--durations=10'  # Show 10 slowest tests
    ])