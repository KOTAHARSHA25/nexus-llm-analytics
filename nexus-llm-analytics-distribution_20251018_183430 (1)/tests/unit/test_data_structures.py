# Unit Tests for Data Structures
# Production-grade unit testing for optimized data structures

import pytest
import time
import threading
import random
import string
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import sys
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import os

# Import the optimized data structures
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_data_structures import (
    OptimizedTrie, 
    HighPerformanceHashMap, 
    OptimizedDataProcessor,
    PerformanceMonitor
)

class TestOptimizedTrie:
    """Comprehensive unit tests for OptimizedTrie"""
    
    @pytest.fixture
    def trie(self):
        """Create fresh trie instance for each test"""
        return OptimizedTrie()
    
    @pytest.fixture
    def populated_trie(self):
        """Create trie with sample data"""
        trie = OptimizedTrie()
        test_words = [
            "hello", "world", "python", "test", "optimization",
            "performance", "data", "structure", "algorithm", "efficient"
        ]
        for word in test_words:
            trie.insert(word, f"value_for_{word}")
        return trie
    
    @pytest.mark.unit
    def test_trie_initialization(self, trie):
        """Test trie proper initialization"""
        assert trie.root is not None
        assert trie.size == 0
        assert trie.get_stats()['total_nodes'] == 1  # root node
        assert not trie.root.is_end_of_word
    
    @pytest.mark.unit
    def test_single_insertion(self, trie):
        """Test inserting single word"""
        result = trie.insert("test", "test_value")
        
        assert result is True
        assert trie.size == 1
        assert trie.search("test") == "test_value"
    
    @pytest.mark.unit
    def test_multiple_insertions(self, trie):
        """Test inserting multiple words"""
        words = ["cat", "car", "card", "care", "careful"]
        
        for i, word in enumerate(words):
            result = trie.insert(word, f"value_{i}")
            assert result is True
            assert trie.size == i + 1
        
        # Verify all words are searchable
        for i, word in enumerate(words):
            assert trie.search(word) == f"value_{i}"
    
    @pytest.mark.unit
    def test_duplicate_insertion(self, trie):
        """Test inserting duplicate words"""
        # First insertion
        result1 = trie.insert("duplicate", "value1")
        assert result1 is True
        assert trie.size == 1
        
        # Duplicate insertion (should update value)
        result2 = trie.insert("duplicate", "value2")
        assert result2 is False  # Not a new insertion
        assert trie.size == 1  # Size unchanged
        assert trie.search("duplicate") == "value2"  # Value updated
    
    @pytest.mark.unit
    def test_search_existing_words(self, populated_trie):
        """Test searching for existing words"""
        assert populated_trie.search("hello") == "value_for_hello"
        assert populated_trie.search("python") == "value_for_python"
        assert populated_trie.search("optimization") == "value_for_optimization"
    
    @pytest.mark.unit
    def test_search_non_existing_words(self, populated_trie):
        """Test searching for non-existing words"""
        assert populated_trie.search("nonexistent") is None
        assert populated_trie.search("") is None
        assert populated_trie.search("hel") is None  # Prefix, not complete word
    
    @pytest.mark.unit
    def test_prefix_search(self, populated_trie):
        """Test prefix-based search functionality"""
        # Add more words with common prefix
        populated_trie.insert("hello_world", "hello_world_value")
        populated_trie.insert("hello_python", "hello_python_value")
        
        prefix_results = populated_trie.get_words_with_prefix("hello")
        
        assert len(prefix_results) == 3
        assert "hello" in prefix_results
        assert "hello_world" in prefix_results
        assert "hello_python" in prefix_results
    
    @pytest.mark.unit
    def test_empty_string_operations(self, trie):
        """Test operations with empty strings"""
        # Insert empty string
        result = trie.insert("", "empty_value")
        assert result is True
        assert trie.size == 1
        assert trie.search("") == "empty_value"
        
        # Prefix search with empty string should return all words
        trie.insert("test", "test_value")
        all_words = trie.get_words_with_prefix("")
        assert len(all_words) == 2
    
    @pytest.mark.unit
    def test_case_sensitivity(self, trie):
        """Test case sensitivity in trie operations"""
        trie.insert("Test", "upper_value")
        trie.insert("test", "lower_value")
        trie.insert("TEST", "all_upper_value")
        
        assert trie.size == 3
        assert trie.search("Test") == "upper_value"
        assert trie.search("test") == "lower_value"
        assert trie.search("TEST") == "all_upper_value"
    
    @pytest.mark.unit
    def test_unicode_support(self, trie):
        """Test unicode character support"""
        unicode_words = ["café", "naïve", "résumé", "你好", "🚀"]
        
        for word in unicode_words:
            result = trie.insert(word, f"value_for_{word}")
            assert result is True
        
        assert trie.size == len(unicode_words)
        
        for word in unicode_words:
            assert trie.search(word) == f"value_for_{word}"
    
    @pytest.mark.unit 
    @pytest.mark.performance
    def test_insertion_performance(self, trie, performance_timer):
        """Test insertion performance with large dataset"""
        # Generate test data
        test_words = []
        for i in range(10000):
            word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 15)))
            test_words.append(word)
        
        # Measure insertion time
        performance_timer.start()
        for word in test_words:
            trie.insert(word, f"value_{word}")
        performance_timer.stop()
        
        # Verify results
        assert trie.size <= len(test_words)  # <= because of potential duplicates
        insertion_time = performance_timer.elapsed
        
        # Performance assertion: should handle 10k insertions in reasonable time
        assert insertion_time < 5.0, f"Insertion took {insertion_time:.2f}s, expected < 5.0s"
        
        # Calculate insertions per second
        ops_per_second = len(test_words) / insertion_time
        assert ops_per_second > 2000, f"Only {ops_per_second:.0f} ops/sec, expected > 2000"
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_search_performance(self, populated_trie, performance_timer):
        """Test search performance"""
        # Add more data for meaningful performance test
        for i in range(1000):
            word = f"performance_test_word_{i}"
            populated_trie.insert(word, f"value_{i}")
        
        # Prepare search terms (mix of existing and non-existing)
        search_terms = []
        for i in range(500):
            search_terms.append(f"performance_test_word_{i}")  # existing
        for i in range(500):
            search_terms.append(f"nonexistent_word_{i}")  # non-existing
        
        # Measure search time
        performance_timer.start()
        for term in search_terms:
            populated_trie.search(term)
        performance_timer.stop()
        
        search_time = performance_timer.elapsed
        ops_per_second = len(search_terms) / search_time
        
        # Performance assertion
        assert ops_per_second > 5000, f"Only {ops_per_second:.0f} searches/sec, expected > 5000"
    
    @pytest.mark.unit
    def test_memory_efficiency(self, trie):
        """Test memory usage efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Insert significant amount of data
        for i in range(10000):
            word = f"memory_test_word_{i}_with_longer_suffix"
            trie.insert(word, f"value_for_word_{i}")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory per word should be reasonable
        memory_per_word = memory_increase / 10000
        assert memory_per_word < 0.001, f"Memory per word: {memory_per_word:.6f}MB, expected < 0.001MB"
    
    @pytest.mark.unit
    def test_thread_safety(self, trie):
        """Test thread safety of trie operations"""
        num_threads = 10
        words_per_thread = 100
        results = []
        
        def insert_words(thread_id):
            thread_results = []
            for i in range(words_per_thread):
                word = f"thread_{thread_id}_word_{i}"
                result = trie.insert(word, f"value_{thread_id}_{i}")
                thread_results.append((word, result))
            return thread_results
        
        # Use ThreadPoolExecutor for controlled concurrent access
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(insert_words, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                results.extend(future.result())
        
        # Verify all insertions completed
        assert len(results) == num_threads * words_per_thread
        assert trie.size <= num_threads * words_per_thread  # <= due to potential race conditions
        
        # Verify data integrity by searching
        successful_insertions = 0
        for word, insert_result in results:
            if trie.search(word) is not None:
                successful_insertions += 1
        
        # Should have high success rate despite potential race conditions
        success_rate = successful_insertions / len(results)
        assert success_rate > 0.95, f"Success rate {success_rate:.2f} too low"
    
    @pytest.mark.unit
    def test_statistics_tracking(self, populated_trie):
        """Test statistics and monitoring functionality"""
        stats = populated_trie.get_stats()
        
        # Verify basic stats structure
        assert 'total_nodes' in stats
        assert 'total_words' in stats
        assert 'max_depth' in stats
        assert 'memory_usage_bytes' in stats
        
        # Verify stats accuracy
        assert stats['total_words'] == populated_trie.size
        assert stats['total_nodes'] >= stats['total_words']  # At least one node per word
        assert stats['max_depth'] > 0
        assert stats['memory_usage_bytes'] > 0
    
    @pytest.mark.unit
    def test_error_handling(self, trie):
        """Test error handling and edge cases"""
        # Test None inputs
        with pytest.raises(TypeError):
            trie.insert(None, "value")
        
        with pytest.raises(TypeError):
            trie.search(None)
        
        # Test non-string inputs
        with pytest.raises(TypeError):
            trie.insert(123, "value")
        
        with pytest.raises(TypeError):
            trie.search(123)


class TestHighPerformanceHashMap:
    """Comprehensive unit tests for HighPerformanceHashMap"""
    
    @pytest.fixture
    def hashmap(self):
        """Create fresh hashmap instance"""
        return HighPerformanceHashMap(initial_capacity=16)
    
    @pytest.fixture
    def populated_hashmap(self):
        """Create hashmap with sample data"""
        hm = HighPerformanceHashMap()
        for i in range(100):
            hm.put(f"key_{i}", f"value_{i}")
        return hm
    
    @pytest.mark.unit
    def test_hashmap_initialization(self, hashmap):
        """Test hashmap proper initialization"""
        assert hashmap.size() == 0
        assert hashmap.capacity > 0
        assert hashmap.load_factor == 0.0
        assert hashmap.is_empty()
    
    @pytest.mark.unit
    def test_basic_put_get_operations(self, hashmap):
        """Test basic put and get operations"""
        # Test putting and getting
        hashmap.put("test_key", "test_value")
        assert hashmap.get("test_key") == "test_value"
        assert hashmap.size() == 1
        assert not hashmap.is_empty()
    
    @pytest.mark.unit
    def test_put_update_existing_key(self, hashmap):
        """Test updating existing key"""
        hashmap.put("key", "value1")
        assert hashmap.size() == 1
        
        hashmap.put("key", "value2")  # Update
        assert hashmap.size() == 1  # Size unchanged
        assert hashmap.get("key") == "value2"  # Value updated
    
    @pytest.mark.unit
    def test_get_nonexistent_key(self, hashmap):
        """Test getting non-existent key"""
        assert hashmap.get("nonexistent") is None
        assert hashmap.get("nonexistent", "default") == "default"
    
    @pytest.mark.unit
    def test_contains_key(self, populated_hashmap):
        """Test key existence checking"""
        assert populated_hashmap.contains_key("key_0")
        assert populated_hashmap.contains_key("key_50")
        assert not populated_hashmap.contains_key("nonexistent_key")
    
    @pytest.mark.unit
    def test_remove_operations(self, populated_hashmap):
        """Test key removal operations"""
        initial_size = populated_hashmap.size()
        
        # Remove existing key
        removed_value = populated_hashmap.remove("key_0")
        assert removed_value == "value_0"
        assert populated_hashmap.size() == initial_size - 1
        assert not populated_hashmap.contains_key("key_0")
        
        # Remove non-existent key
        removed_value = populated_hashmap.remove("nonexistent")
        assert removed_value is None
        assert populated_hashmap.size() == initial_size - 1
    
    @pytest.mark.unit
    def test_clear_operation(self, populated_hashmap):
        """Test clearing hashmap"""
        assert populated_hashmap.size() > 0
        
        populated_hashmap.clear()
        
        assert populated_hashmap.size() == 0
        assert populated_hashmap.is_empty()
        assert not populated_hashmap.contains_key("key_0")
    
    @pytest.mark.unit
    def test_keys_values_items(self, populated_hashmap):
        """Test getting keys, values, and items"""
        keys = populated_hashmap.keys()
        values = populated_hashmap.values()
        items = populated_hashmap.items()
        
        assert len(keys) == populated_hashmap.size()
        assert len(values) == populated_hashmap.size()
        assert len(items) == populated_hashmap.size()
        
        # Verify consistency
        for key, value in items:
            assert key in keys
            assert value in values
            assert populated_hashmap.get(key) == value
    
    @pytest.mark.unit
    def test_load_factor_and_resize(self):
        """Test load factor calculation and automatic resizing"""
        hm = HighPerformanceHashMap(initial_capacity=8, load_factor_threshold=0.75)
        initial_capacity = hm.capacity
        
        # Fill beyond load factor threshold
        num_items = int(initial_capacity * 0.8)  # Exceed threshold
        for i in range(num_items):
            hm.put(f"key_{i}", f"value_{i}")
        
        # Should have triggered resize
        assert hm.capacity > initial_capacity
        assert hm.load_factor <= 0.75
        
        # Verify all data integrity after resize
        for i in range(num_items):
            assert hm.get(f"key_{i}") == f"value_{i}"
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_performance_large_dataset(self, performance_timer):
        """Test performance with large dataset"""
        hm = HighPerformanceHashMap()
        num_items = 50000
        
        # Test insertion performance
        performance_timer.start()
        for i in range(num_items):
            hm.put(f"performance_key_{i}", f"performance_value_{i}")
        performance_timer.stop()
        
        insertion_time = performance_timer.elapsed
        insertions_per_second = num_items / insertion_time
        
        assert insertions_per_second > 10000, f"Only {insertions_per_second:.0f} insertions/sec"
        
        # Test lookup performance
        performance_timer.start()
        for i in range(0, num_items, 10):  # Sample every 10th item
            value = hm.get(f"performance_key_{i}")
            assert value == f"performance_value_{i}"
        performance_timer.stop()
        
        lookup_time = performance_timer.elapsed
        lookups_per_second = (num_items // 10) / lookup_time
        
        assert lookups_per_second > 50000, f"Only {lookups_per_second:.0f} lookups/sec"
    
    @pytest.mark.unit
    def test_hash_collision_handling(self):
        """Test handling of hash collisions"""
        # Create hashmap with small capacity to force collisions
        hm = HighPerformanceHashMap(initial_capacity=4)
        
        # Insert items that will likely collide
        collision_keys = []
        for i in range(20):
            key = f"collision_test_{i}"
            collision_keys.append(key)
            hm.put(key, f"value_{i}")
        
        assert hm.size() == 20
        
        # Verify all values can be retrieved correctly
        for i, key in enumerate(collision_keys):
            assert hm.get(key) == f"value_{i}"
    
    @pytest.mark.unit
    def test_thread_safety(self):
        """Test thread safety of hashmap operations"""
        hm = HighPerformanceHashMap()
        num_threads = 8
        items_per_thread = 1000
        
        def worker_thread(thread_id):
            results = []
            for i in range(items_per_thread):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                hm.put(key, value)
                
                # Verify immediate retrieval
                retrieved = hm.get(key)
                results.append(retrieved == value)
            return results
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Verify high success rate
        success_rate = sum(all_results) / len(all_results)
        assert success_rate > 0.95, f"Success rate {success_rate:.2f} too low"
        
        # Verify final state
        assert hm.size() <= num_threads * items_per_thread
    
    @pytest.mark.unit
    def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        hm = HighPerformanceHashMap()
        
        # Insert data
        for i in range(10000):
            hm.put(f"memory_test_key_{i}", f"memory_test_value_{i}_with_extra_data")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory per item should be reasonable
        memory_per_item = memory_increase / 10000
        assert memory_per_item < 0.002, f"Memory per item: {memory_per_item:.6f}MB"
    
    @pytest.mark.unit
    def test_statistics_and_metrics(self, populated_hashmap):
        """Test statistics and performance metrics"""
        stats = populated_hashmap.get_performance_metrics()
        
        # Verify stats structure
        expected_keys = ['size', 'capacity', 'load_factor', 'collision_count', 'resize_count']
        for key in expected_keys:
            assert key in stats
        
        # Verify stats accuracy
        assert stats['size'] == populated_hashmap.size()
        assert stats['capacity'] == populated_hashmap.capacity
        assert 0 <= stats['load_factor'] <= 1
        assert stats['collision_count'] >= 0
        assert stats['resize_count'] >= 0
    
    @pytest.mark.unit
    def test_error_handling(self, hashmap):
        """Test error handling and edge cases"""
        # Test None key
        with pytest.raises(TypeError):
            hashmap.put(None, "value")
        
        with pytest.raises(TypeError):
            hashmap.get(None)
        
        with pytest.raises(TypeError):
            hashmap.contains_key(None)
        
        with pytest.raises(TypeError):
            hashmap.remove(None)


class TestOptimizedDataProcessor:
    """Comprehensive unit tests for OptimizedDataProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create fresh data processor"""
        return OptimizedDataProcessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return {
            'users': [
                {'id': 1, 'name': 'John', 'age': 30, 'city': 'New York'},
                {'id': 2, 'name': 'Jane', 'age': 25, 'city': 'Los Angeles'},
                {'id': 3, 'name': 'Bob', 'age': 35, 'city': 'Chicago'}
            ],
            'metadata': {
                'total_count': 3,
                'version': '1.0'
            }
        }
    
    @pytest.mark.unit
    def test_processor_initialization(self, processor):
        """Test processor proper initialization"""
        assert processor is not None
        assert hasattr(processor, 'process_data')
        assert hasattr(processor, 'get_performance_metrics')
    
    @pytest.mark.unit
    def test_process_simple_data(self, processor, sample_data):
        """Test processing simple data structure"""
        result = processor.process_data(sample_data)
        
        assert result is not None
        assert 'processed_data' in result
        assert 'metadata' in result
        assert result['metadata']['processing_time'] > 0
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_parallel_processing_performance(self, processor, performance_timer):
        """Test parallel processing performance"""
        # Create larger dataset
        large_data = []
        for i in range(10000):
            large_data.append({
                'id': i,
                'value': i * 2,
                'description': f'Item {i} description'
            })
        
        # Process with parallel processing
        performance_timer.start()
        result = processor.process_data_parallel(large_data, num_workers=4)
        performance_timer.stop()
        
        parallel_time = performance_timer.elapsed
        
        # Process sequentially for comparison
        performance_timer.start()
        sequential_result = processor.process_data(large_data)
        performance_timer.stop()
        
        sequential_time = performance_timer.elapsed
        
        # Parallel should be faster for large datasets
        if sequential_time > 1.0:  # Only assert if sequential takes meaningful time
            speedup = sequential_time / parallel_time
            assert speedup > 1.5, f"Parallel speedup {speedup:.2f}x insufficient"
    
    @pytest.mark.unit
    def test_error_handling_invalid_data(self, processor):
        """Test error handling with invalid data"""
        # Test None data
        result = processor.process_data(None)
        assert result['error'] is not None
        
        # Test empty data
        result = processor.process_data({})
        assert 'processed_data' in result
        
        # Test malformed data
        malformed_data = {'invalid': 'structure', 'missing': None}
        result = processor.process_data(malformed_data)
        assert result is not None  # Should handle gracefully
    
    @pytest.mark.unit
    def test_memory_cleanup(self, processor):
        """Test proper memory cleanup after processing"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large amount of data
        for i in range(10):
            large_data = [{'id': j, 'data': f'data_{j}' * 100} for j in range(1000)]
            result = processor.process_data(large_data)
            del result
            del large_data
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal after cleanup
        assert memory_increase < 50, f"Memory leak detected: {memory_increase:.2f}MB increase"


class TestPerformanceMonitor:
    """Unit tests for PerformanceMonitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create fresh performance monitor"""
        return PerformanceMonitor()
    
    @pytest.mark.unit
    def test_monitor_initialization(self, monitor):
        """Test monitor proper initialization"""
        assert monitor is not None
        metrics = monitor.get_metrics()
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'operation_counts' in metrics
    
    @pytest.mark.unit
    def test_operation_tracking(self, monitor):
        """Test operation tracking functionality"""
        # Record some operations
        monitor.record_operation('insert', 0.1)
        monitor.record_operation('search', 0.05)
        monitor.record_operation('insert', 0.12)
        
        metrics = monitor.get_metrics()
        
        assert metrics['operation_counts']['insert'] == 2
        assert metrics['operation_counts']['search'] == 1
        assert 'average_times' in metrics
        assert metrics['average_times']['insert'] > 0
        assert metrics['average_times']['search'] > 0
    
    @pytest.mark.unit
    def test_resource_monitoring(self, monitor):
        """Test system resource monitoring"""
        metrics = monitor.get_metrics()
        
        # CPU usage should be reasonable
        assert 0 <= metrics['cpu_usage'] <= 100
        
        # Memory usage should be positive
        assert metrics['memory_usage'] > 0
        
        # Should have timestamps
        assert 'timestamp' in metrics
    
    @pytest.mark.unit
    def test_performance_alerts(self, monitor):
        """Test performance alerting system"""
        # Record slow operations
        monitor.record_operation('slow_operation', 5.0)
        
        alerts = monitor.get_alerts()
        
        # Should generate alert for slow operation
        assert len(alerts) > 0
        slow_alert = next((a for a in alerts if 'slow_operation' in a['message']), None)
        assert slow_alert is not None
        assert slow_alert['severity'] in ['warning', 'critical']


# Integration tests combining multiple components
class TestDataStructureIntegration:
    """Integration tests for data structure components"""
    
    @pytest.mark.integration
    def test_trie_hashmap_integration(self):
        """Test integration between trie and hashmap"""
        trie = OptimizedTrie()
        hashmap = HighPerformanceHashMap()
        
        # Use trie for prefix search, hashmap for fast lookup
        test_data = {
            'apple': 'fruit',
            'application': 'software',
            'apply': 'action',
            'banana': 'fruit',
            'band': 'music'
        }
        
        # Populate both structures
        for word, category in test_data.items():
            trie.insert(word, word)  # Store word for prefix search
            hashmap.put(word, category)  # Store category for fast lookup
        
        # Test integration: find all words with prefix, then get categories
        apple_words = trie.get_words_with_prefix('app')
        categories = [hashmap.get(word) for word in apple_words if hashmap.contains_key(word)]
        
        assert len(apple_words) == 3
        assert 'fruit' in categories
        assert 'software' in categories
        assert 'action' in categories
    
    @pytest.mark.integration
    def test_processor_with_data_structures(self):
        """Test data processor using optimized data structures"""
        processor = OptimizedDataProcessor()
        
        # Create test data that would benefit from optimized structures
        test_data = {
            'search_index': ['apple', 'application', 'apply', 'banana', 'band'],
            'metadata': {
                'apple': {'type': 'fruit', 'color': 'red'},
                'application': {'type': 'software', 'platform': 'web'},
                'apply': {'type': 'action', 'category': 'verb'},
                'banana': {'type': 'fruit', 'color': 'yellow'},
                'band': {'type': 'music', 'genre': 'rock'}
            }
        }
        
        result = processor.process_data(test_data)
        
        assert result is not None
        assert 'processed_data' in result
        assert result['metadata']['processing_time'] > 0
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_end_to_end_performance(self, performance_timer):
        """Test end-to-end performance with all components"""
        # Initialize all components
        trie = OptimizedTrie()
        hashmap = HighPerformanceHashMap()
        processor = OptimizedDataProcessor()
        monitor = PerformanceMonitor()
        
        # Generate realistic test data
        num_items = 10000
        test_words = [f'word_{i}_{random.choice(["apple", "banana", "cherry"])}' 
                     for i in range(num_items)]
        
        performance_timer.start()
        
        # Simulate realistic workflow
        for word in test_words:
            # Insert into trie for search
            trie.insert(word, f'data_for_{word}')
            
            # Store metadata in hashmap
            hashmap.put(word, {'category': 'test', 'index': word.split('_')[1]})
            
            # Record operation
            monitor.record_operation('data_insertion', 0.001)
        
        # Process some prefix searches
        for prefix in ['word_100', 'word_200', 'word_300']:
            found_words = trie.get_words_with_prefix(prefix)
            for word in found_words:
                metadata = hashmap.get(word)
                monitor.record_operation('search_operation', 0.0005)
        
        performance_timer.stop()
        
        # Verify performance
        total_time = performance_timer.elapsed
        operations_per_second = (num_items + 300) / total_time  # Approximate total operations
        
        assert operations_per_second > 1000, f"Only {operations_per_second:.0f} ops/sec"
        
        # Verify data integrity
        assert trie.size > 0
        assert hashmap.size() > 0
        
        # Check monitoring data
        metrics = monitor.get_metrics()
        assert metrics['operation_counts']['data_insertion'] == num_items
        assert metrics['operation_counts']['search_operation'] > 0


if __name__ == '__main__':
    # Run specific test categories
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--cov=src.backend.core.optimized_data_structures',
        '--cov-report=term-missing'
    ])