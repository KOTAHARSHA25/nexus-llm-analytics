# Stress Testing and Robustness Validation
# Comprehensive stress tests for system robustness under extreme conditions

import pytest
import asyncio
import threading
import time
import random
import gc
import sys
import os
import tempfile
import json
import psutil
from typing import List, Dict, Any, Optional, Callable
from unittest.mock import patch, Mock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Import components for stress testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_data_structures import OptimizedTrie, HighPerformanceHashMap, OptimizedDataProcessor
from backend.core.optimized_llm_client import OptimizedLLMClient, SmartCache, OptimizedConnectionPool
from backend.core.rate_limiter import RateLimiter
from backend.core.optimized_file_io import OptimizedFileProcessor, StreamingCSVReader
from backend.core.enhanced_cache_integration import EnhancedCacheManager
from backend.core.intelligent_query_engine import IntelligentQueryRouter, QueryPatternAnalyzer


class StressTestMetrics:
    """Utility for tracking stress test metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.operations_completed = 0
        self.operations_failed = 0
        self.peak_memory_mb = 0
        self.errors = []
        self.performance_samples = []
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        gc.collect()  # Clean start
    
    def end(self):
        """End timing and calculate metrics"""
        self.end_time = time.time()
        self.peak_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        gc.collect()
    
    def record_operation(self, success: bool, duration: float = None, error: Exception = None):
        """Record operation result"""
        if success:
            self.operations_completed += 1
        else:
            self.operations_failed += 1
            if error:
                self.errors.append(str(error))
        
        if duration:
            self.performance_samples.append(duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        total_ops = self.operations_completed + self.operations_failed
        success_rate = self.operations_completed / total_ops if total_ops > 0 else 0
        ops_per_second = total_ops / total_time if total_time > 0 else 0
        
        avg_duration = sum(self.performance_samples) / len(self.performance_samples) if self.performance_samples else 0
        
        return {
            'total_time_seconds': total_time,
            'operations_completed': self.operations_completed,
            'operations_failed': self.operations_failed,
            'success_rate': success_rate,
            'operations_per_second': ops_per_second,
            'peak_memory_mb': self.peak_memory_mb,
            'average_operation_duration': avg_duration,
            'error_count': len(self.errors),
            'unique_errors': len(set(self.errors)),
        }


class TestDataStructureStress:
    """Stress tests for optimized data structures"""
    
    def test_trie_high_volume_insertions(self):
        """Test trie with high volume of insertions"""
        trie = OptimizedTrie()
        metrics = StressTestMetrics()
        metrics.start()
        
        # Generate test data
        test_size = 50000  # Reduced from 100k for faster testing
        test_data = []
        for i in range(test_size):
            key = f"stress_key_{i}_{random.randint(1000, 9999)}"
            value = f"stress_value_{i}_{hash(key) % 10000}"
            test_data.append((key, value))
        
        # Insert data
        for key, value in test_data:
            try:
                start_op = time.time()
                trie.insert(key, value)
                end_op = time.time()
                metrics.record_operation(True, end_op - start_op)
            except Exception as e:
                metrics.record_operation(False, error=e)
        
        metrics.end()
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] > 0.99, f"Too many insertion failures: {summary['success_rate']:.2%}"
        assert summary['operations_per_second'] > 1000, f"Too slow: {summary['operations_per_second']:.0f} ops/sec"
        assert trie.size == metrics.operations_completed, "Size mismatch after insertions"
        
        print(f"Trie stress test: {summary}")
    
    def test_hashmap_concurrent_stress(self):
        """Test hashmap under concurrent stress"""
        hashmap = HighPerformanceHashMap()
        metrics = StressTestMetrics()
        metrics.start()
        
        def worker_thread(thread_id: int, operations_per_thread: int):
            """Worker thread performing operations"""
            local_errors = []
            local_completed = 0
            
            for i in range(operations_per_thread):
                try:
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    
                    # Mix of operations
                    operation = random.choice(['put', 'get', 'contains', 'remove'])
                    
                    if operation == 'put':
                        hashmap.put(key, value)
                        local_completed += 1
                    elif operation == 'get':
                        result = hashmap.get(key)
                        local_completed += 1
                    elif operation == 'contains':
                        exists = hashmap.contains_key(key)
                        local_completed += 1
                    elif operation == 'remove':
                        hashmap.remove(key)
                        local_completed += 1
                        
                except Exception as e:
                    local_errors.append(str(e))
            
            return local_completed, local_errors
        
        # Run concurrent threads
        num_threads = 10
        operations_per_thread = 5000
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for thread_id in range(num_threads):
                future = executor.submit(worker_thread, thread_id, operations_per_thread)
                futures.append(future)
            
            # Collect results
            for future in futures:
                completed, errors = future.result()
                metrics.operations_completed += completed
                metrics.operations_failed += len(errors)
                metrics.errors.extend(errors)
        
        metrics.end()
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] > 0.95, f"Too many concurrent failures: {summary['success_rate']:.2%}"
        assert summary['operations_per_second'] > 5000, f"Too slow under concurrency: {summary['operations_per_second']:.0f} ops/sec"
        
        print(f"HashMap concurrent stress test: {summary}")
    
    def test_data_processor_memory_pressure(self):
        """Test data processor under memory pressure"""
        processor = OptimizedDataProcessor()
        metrics = StressTestMetrics()
        metrics.start()
        
        # Generate large datasets
        large_datasets = []
        for i in range(10):  # Reduced from 20
            dataset = {
                'id': i,
                'data': [random.randint(1, 1000) for _ in range(10000)],  # Reduced from 50k
                'metadata': {
                    'created_at': time.time(),
                    'large_text': 'x' * 10000,  # 10KB text
                    'nested_data': {'level_' + str(j): f'value_{j}' for j in range(100)}
                }
            }
            large_datasets.append(dataset)
        
        # Process datasets
        for i, dataset in enumerate(large_datasets):
            try:
                start_op = time.time()
                result = asyncio.run(processor.process_data(dataset))
                end_op = time.time()
                
                if result and 'error' not in result:
                    metrics.record_operation(True, end_op - start_op)
                else:
                    metrics.record_operation(False, error=Exception(f"Processing failed for dataset {i}"))
                    
            except Exception as e:
                metrics.record_operation(False, error=e)
        
        metrics.end()
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] > 0.8, f"Too many processing failures: {summary['success_rate']:.2%}"
        assert summary['peak_memory_mb'] < 1000, f"Memory usage too high: {summary['peak_memory_mb']:.0f}MB"
        
        print(f"Data processor memory pressure test: {summary}")


class TestCacheStress:
    """Stress tests for caching systems"""
    
    def test_cache_high_frequency_access(self):
        """Test cache with high frequency access patterns"""
        cache = SmartCache(max_size=10000, ttl=300)
        metrics = StressTestMetrics()
        metrics.start()
        
        # Generate test data with realistic access patterns
        hot_keys = [f"hot_key_{i}" for i in range(100)]  # 1% hot data
        warm_keys = [f"warm_key_{i}" for i in range(1000)]  # 10% warm data
        cold_keys = [f"cold_key_{i}" for i in range(9000)]  # 90% cold data
        
        all_keys = hot_keys + warm_keys + cold_keys
        
        # Initial population
        for key in all_keys[:5000]:  # Don't exceed cache size initially
            try:
                cache.put(key, f"value_for_{key}")
                metrics.record_operation(True)
            except Exception as e:
                metrics.record_operation(False, error=e)
        
        # High frequency access simulation
        for _ in range(50000):  # 50k operations
            try:
                start_op = time.time()
                
                # Realistic access pattern: 80% hot, 15% warm, 5% cold
                rand = random.random()
                if rand < 0.8:
                    key = random.choice(hot_keys)
                elif rand < 0.95:
                    key = random.choice(warm_keys)
                else:
                    key = random.choice(cold_keys)
                
                # Mix of get and put operations
                if random.random() < 0.7:  # 70% reads
                    result = cache.get(key)
                    if result is None and key in all_keys[:5000]:
                        # Cache miss for initially loaded key might indicate eviction
                        pass
                else:  # 30% writes
                    cache.put(key, f"updated_value_for_{key}_{time.time()}")
                
                end_op = time.time()
                metrics.record_operation(True, end_op - start_op)
                
            except Exception as e:
                metrics.record_operation(False, error=e)
        
        metrics.end()
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] > 0.99, f"Too many cache failures: {summary['success_rate']:.2%}"
        assert summary['operations_per_second'] > 10000, f"Cache too slow: {summary['operations_per_second']:.0f} ops/sec"
        
        print(f"Cache high frequency stress test: {summary}")
    
    def test_cache_ttl_expiration_stress(self):
        """Test cache TTL expiration under stress"""
        cache = SmartCache(max_size=1000, ttl=0.1)  # 100ms TTL
        metrics = StressTestMetrics()
        metrics.start()
        
        # Rapid insertion and expiration
        for batch in range(100):  # 100 batches
            batch_keys = []
            
            # Insert batch
            for i in range(50):  # 50 items per batch
                key = f"batch_{batch}_key_{i}"
                value = f"batch_{batch}_value_{i}"
                batch_keys.append(key)
                
                try:
                    cache.put(key, value)
                    metrics.record_operation(True)
                except Exception as e:
                    metrics.record_operation(False, error=e)
            
            # Wait for partial expiration
            time.sleep(0.05)  # 50ms
            
            # Try to access some keys (some might be expired)
            for key in batch_keys[:25]:  # Check half
                try:
                    result = cache.get(key)
                    metrics.record_operation(True)
                except Exception as e:
                    metrics.record_operation(False, error=e)
            
            # Wait for full expiration
            time.sleep(0.1)  # Another 100ms
            
            # Check expiration (should be mostly expired)
            expired_count = 0
            for key in batch_keys:
                try:
                    result = cache.get(key)
                    if result is None:
                        expired_count += 1
                    metrics.record_operation(True)
                except Exception as e:
                    metrics.record_operation(False, error=e)
            
            # Most items should be expired
            expiration_rate = expired_count / len(batch_keys)
            assert expiration_rate > 0.8, f"TTL expiration not working properly: {expiration_rate:.2%} expired"
        
        metrics.end()
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] > 0.95, f"Too many TTL failures: {summary['success_rate']:.2%}"
        
        print(f"Cache TTL expiration stress test: {summary}")


class TestFileIOStress:
    """Stress tests for file I/O operations"""
    
    def test_concurrent_file_processing_stress(self):
        """Test concurrent file processing under stress"""
        processor = OptimizedFileProcessor()
        metrics = StressTestMetrics()
        metrics.start()
        
        # Create test files
        test_files = []
        file_types = ['json', 'csv', 'txt']
        
        for i in range(30):  # Reduced from 50
            file_type = random.choice(file_types)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{file_type}', delete=False) as f:
                if file_type == 'json':
                    data = {
                        'id': i,
                        'items': [{'item_id': j, 'value': random.randint(1, 1000)} for j in range(100)],
                        'metadata': {'created': time.time(), 'type': 'test'}
                    }
                    json.dump(data, f)
                elif file_type == 'csv':
                    f.write('id,name,value,timestamp\n')
                    for j in range(1000):  # Reduced from 5000
                        f.write(f'{j},item_{j},{random.randint(1, 1000)},{time.time()}\n')
                else:  # txt
                    for j in range(1000):
                        f.write(f'Line {j}: Some test content with random number {random.randint(1, 1000)}\n')
                
                test_files.append(f.name)
        
        def process_file_worker(file_path: str) -> tuple:
            """Worker function to process a single file"""
            try:
                start_op = time.time()
                result = asyncio.run(processor.process_file(file_path))
                end_op = time.time()
                
                if 'error' not in result:
                    return True, end_op - start_op, None
                else:
                    return False, end_op - start_op, Exception(result['error'])
            except Exception as e:
                return False, 0, e
        
        # Process files concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for file_path in test_files:
                future = executor.submit(process_file_worker, file_path)
                futures.append(future)
            
            # Collect results
            for future in futures:
                success, duration, error = future.result()
                metrics.record_operation(success, duration, error)
        
        # Clean up
        for file_path in test_files:
            try:
                os.unlink(file_path)
            except OSError:
                pass
        
        metrics.end()
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] > 0.9, f"Too many file processing failures: {summary['success_rate']:.2%}"
        assert summary['average_operation_duration'] < 1.0, f"File processing too slow: {summary['average_operation_duration']:.2f}s avg"
        
        print(f"Concurrent file processing stress test: {summary}")
    
    def test_large_file_streaming_stress(self):
        """Test streaming of large files under stress"""
        metrics = StressTestMetrics()
        metrics.start()
        
        # Create large CSV files
        large_files = []
        for i in range(5):  # 5 large files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                # Write header
                f.write('id,name,category,value,timestamp,description\n')
                
                # Write large amount of data
                for j in range(50000):  # 50k rows per file
                    row = f'{j},item_{j},category_{j%10},{random.randint(1, 10000)},{time.time()},Description for item {j} with some text\n'
                    f.write(row)
                
                large_files.append(f.name)
        
        # Stream process each file
        for file_path in large_files:
            try:
                start_op = time.time()
                
                reader = StreamingCSVReader(file_path)
                row_count = 0
                
                async def stream_process():
                    nonlocal row_count
                    async for batch in reader.read_batches(batch_size=1000):
                        row_count += len(batch)
                        # Simulate processing
                        await asyncio.sleep(0.001)  # 1ms processing per batch
                
                asyncio.run(stream_process())
                
                end_op = time.time()
                
                if row_count > 40000:  # Should process most rows
                    metrics.record_operation(True, end_op - start_op)
                else:
                    metrics.record_operation(False, error=Exception(f"Too few rows processed: {row_count}"))
                    
            except Exception as e:
                metrics.record_operation(False, error=e)
        
        # Clean up
        for file_path in large_files:
            try:
                os.unlink(file_path)
            except OSError:
                pass
        
        metrics.end()
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] > 0.8, f"Too many streaming failures: {summary['success_rate']:.2%}"
        assert summary['peak_memory_mb'] < 500, f"Memory usage too high during streaming: {summary['peak_memory_mb']:.0f}MB"
        
        print(f"Large file streaming stress test: {summary}")


class TestConnectionStress:
    """Stress tests for connection management"""
    
    def test_connection_pool_exhaustion_recovery(self):
        """Test connection pool under exhaustion and recovery"""
        pool = OptimizedConnectionPool(max_connections=10, timeout=1.0)
        metrics = StressTestMetrics()
        metrics.start()
        
        async def connection_worker(worker_id: int, operations: int):
            """Worker that acquires/releases connections"""
            local_success = 0
            local_failures = 0
            
            for i in range(operations):
                try:
                    # Acquire connection
                    conn = await pool.acquire()
                    
                    if conn:
                        # Simulate work
                        await asyncio.sleep(random.uniform(0.01, 0.1))  # 10-100ms work
                        
                        # Release connection
                        await pool.release(conn)
                        local_success += 1
                    else:
                        local_failures += 1
                        
                except Exception as e:
                    local_failures += 1
            
            return local_success, local_failures
        
        async def run_stress_test():
            # Run multiple workers concurrently
            tasks = []
            num_workers = 20  # More workers than connections
            operations_per_worker = 100
            
            for worker_id in range(num_workers):
                task = asyncio.create_task(connection_worker(worker_id, operations_per_worker))
                tasks.append(task)
            
            # Collect results
            results = await asyncio.gather(*tasks)
            
            total_success = sum(success for success, failures in results)
            total_failures = sum(failures for success, failures in results)
            
            return total_success, total_failures
        
        success_count, failure_count = asyncio.run(run_stress_test())
        
        metrics.operations_completed = success_count
        metrics.operations_failed = failure_count
        metrics.end()
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] > 0.8, f"Too many connection failures: {summary['success_rate']:.2%}"
        
        print(f"Connection pool stress test: {summary}")
    
    def test_rate_limiter_burst_handling(self):
        """Test rate limiter under burst traffic"""
        limiter = RateLimiter(requests_per_second=100, window_size=1.0)
        metrics = StressTestMetrics()
        metrics.start()
        
        async def burst_worker(worker_id: int, burst_size: int):
            """Generate burst traffic"""
            local_allowed = 0
            local_rejected = 0
            
            for i in range(burst_size):
                try:
                    allowed = await limiter.is_allowed(f"client_{worker_id}")
                    if allowed:
                        local_allowed += 1
                    else:
                        local_rejected += 1
                except Exception as e:
                    local_rejected += 1
            
            return local_allowed, local_rejected
        
        async def run_burst_test():
            # Generate multiple bursts
            all_tasks = []
            
            for burst in range(10):  # 10 bursts
                # Create burst of workers
                burst_tasks = []
                for worker_id in range(20):  # 20 workers per burst
                    task = asyncio.create_task(burst_worker(worker_id, 50))  # 50 requests per worker
                    burst_tasks.append(task)
                
                all_tasks.extend(burst_tasks)
                
                # Wait between bursts
                if burst < 9:  # Don't wait after last burst
                    await asyncio.sleep(0.5)  # 500ms between bursts
            
            # Collect all results
            results = await asyncio.gather(*all_tasks)
            
            total_allowed = sum(allowed for allowed, rejected in results)
            total_rejected = sum(rejected for allowed, rejected in results)
            
            return total_allowed, total_rejected
        
        allowed_count, rejected_count = asyncio.run(run_burst_test())
        
        metrics.operations_completed = allowed_count
        metrics.operations_failed = rejected_count
        metrics.end()
        summary = metrics.get_summary()
        
        # Rate limiter should reject some requests during bursts
        total_requests = allowed_count + rejected_count
        rejection_rate = rejected_count / total_requests if total_requests > 0 else 0
        
        # Should allow reasonable throughput but reject bursts
        assert 0.3 < rejection_rate < 0.8, f"Rate limiter rejection rate seems off: {rejection_rate:.2%}"
        assert allowed_count > 500, f"Too few requests allowed: {allowed_count}"
        
        print(f"Rate limiter burst stress test: allowed={allowed_count}, rejected={rejected_count}, rate={rejection_rate:.2%}")


class TestSystemStress:
    """System-wide stress tests"""
    
    def test_end_to_end_system_stress(self):
        """End-to-end system stress test"""
        # Initialize all components
        trie = OptimizedTrie()
        hashmap = HighPerformanceHashMap()
        cache = SmartCache(max_size=1000, ttl=300)
        processor = OptimizedFileProcessor()
        
        metrics = StressTestMetrics()
        metrics.start()
        
        # Create mixed workload
        def mixed_workload_worker(worker_id: int, operations: int):
            """Worker performing mixed operations across all components"""
            local_success = 0
            local_failures = 0
            
            for i in range(operations):
                try:
                    operation_type = random.choice(['trie', 'hashmap', 'cache', 'file'])
                    
                    if operation_type == 'trie':
                        key = f"trie_worker_{worker_id}_key_{i}"
                        value = f"trie_worker_{worker_id}_value_{i}"
                        trie.insert(key, value)
                        result = trie.search(key)
                        assert result == value
                        
                    elif operation_type == 'hashmap':
                        key = f"hash_worker_{worker_id}_key_{i}"
                        value = f"hash_worker_{worker_id}_value_{i}"
                        hashmap.put(key, value)
                        result = hashmap.get(key)
                        assert result == value
                        
                    elif operation_type == 'cache':
                        key = f"cache_worker_{worker_id}_key_{i}"
                        value = f"cache_worker_{worker_id}_value_{i}"
                        cache.put(key, value)
                        result = cache.get(key)
                        # Cache might evict, so None is acceptable
                        if result is not None:
                            assert result == value
                            
                    elif operation_type == 'file':
                        # Create small temp file and process it
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            data = {'worker': worker_id, 'operation': i, 'timestamp': time.time()}
                            json.dump(data, f)
                            temp_path = f.name
                        
                        try:
                            result = asyncio.run(processor.process_file(temp_path))
                            assert 'error' not in result
                        finally:
                            os.unlink(temp_path)
                    
                    local_success += 1
                    
                except Exception as e:
                    local_failures += 1
            
            return local_success, local_failures
        
        # Run mixed workload with multiple threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            for worker_id in range(8):
                future = executor.submit(mixed_workload_worker, worker_id, 500)  # 500 ops per worker
                futures.append(future)
            
            # Collect results
            for future in futures:
                success, failures = future.result()
                metrics.operations_completed += success
                metrics.operations_failed += failures
        
        metrics.end()
        summary = metrics.get_summary()
        
        # System should handle mixed workload well
        assert summary['success_rate'] > 0.95, f"System stress test failure rate too high: {summary['success_rate']:.2%}"
        assert summary['operations_per_second'] > 1000, f"System too slow under stress: {summary['operations_per_second']:.0f} ops/sec"
        
        print(f"End-to-end system stress test: {summary}")
    
    def test_resource_exhaustion_recovery(self):
        """Test system recovery from resource exhaustion"""
        components = {
            'trie': OptimizedTrie(),
            'hashmap': HighPerformanceHashMap(),
            'cache': SmartCache(max_size=100, ttl=60),
        }
        
        metrics = StressTestMetrics()
        metrics.start()
        
        # Phase 1: Normal operation
        for i in range(1000):
            for name, component in components.items():
                try:
                    key = f"{name}_{i}"
                    value = f"value_{i}"
                    
                    if hasattr(component, 'insert'):
                        component.insert(key, value)
                        result = component.search(key)
                        assert result == value
                    elif hasattr(component, 'put'):
                        component.put(key, value)
                        result = component.get(key)
                        if result is not None:  # Cache might evict
                            assert result == value
                    
                    metrics.record_operation(True)
                except Exception as e:
                    metrics.record_operation(False, error=e)
        
        # Phase 2: Resource exhaustion simulation
        try:
            # Try to exhaust memory with large allocations
            large_data = []
            for i in range(1000):  # Reduced to prevent system impact
                large_item = "x" * 10000  # 10KB items
                large_data.append(large_item)
                
                # Try operations during memory pressure
                for name, component in components.items():
                    try:
                        key = f"stress_{name}_{i}"
                        if hasattr(component, 'insert'):
                            component.insert(key, large_item)
                        elif hasattr(component, 'put'):
                            component.put(key, large_item)
                        metrics.record_operation(True)
                    except (MemoryError, OverflowError):
                        # Expected during resource exhaustion
                        pass
                    except Exception as e:
                        metrics.record_operation(False, error=e)
                        
        except MemoryError:
            # Expected when simulating exhaustion
            pass
        
        # Phase 3: Recovery verification
        gc.collect()  # Force garbage collection
        time.sleep(0.1)  # Allow system to recover
        
        # Test that components are still functional
        for name, component in components.items():
            try:
                recovery_key = f"recovery_{name}"
                recovery_value = f"recovery_value_{name}"
                
                if hasattr(component, 'insert'):
                    component.insert(recovery_key, recovery_value)
                    result = component.search(recovery_key)
                    assert result == recovery_value, f"{name} failed to recover"
                elif hasattr(component, 'put'):
                    component.put(recovery_key, recovery_value)
                    result = component.get(recovery_key)
                    if result is not None:  # Cache might still be under pressure
                        assert result == recovery_value, f"{name} failed to recover"
                
                metrics.record_operation(True)
            except Exception as e:
                metrics.record_operation(False, error=e)
                pytest.fail(f"Component {name} failed to recover: {e}")
        
        metrics.end()
        summary = metrics.get_summary()
        
        print(f"Resource exhaustion recovery test: {summary}")


if __name__ == '__main__':
    # Run stress tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-s',  # Show print output
        '--durations=10',  # Show slowest 10 tests
    ])