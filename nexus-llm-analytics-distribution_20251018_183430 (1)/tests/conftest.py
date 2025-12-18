# Comprehensive Test Suite Configuration
# Production-ready testing setup for Nexus LLM Analytics

import pytest
import asyncio
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
import json
import time
import threading
from dataclasses import dataclass
from contextlib import asynccontextmanager
import weakref

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestFixture:
    """Base test fixture configuration"""
    name: str
    description: str
    setup_data: Dict[str, Any]
    cleanup_required: bool = True

class TestDataManager:
    """Centralized test data management"""
    
    def __init__(self):
        self.temp_dir = None
        self.test_files = []
        self.mock_objects = []
        
    def setup_temp_directory(self) -> Path:
        """Create temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp(prefix="nexus_test_")
        return Path(self.temp_dir)
    
    def create_test_csv(self, filename: str, rows: int = 100, columns: List[str] = None) -> Path:
        """Create test CSV file with sample data"""
        if not self.temp_dir:
            self.setup_temp_directory()
            
        columns = columns or ['id', 'name', 'value', 'category', 'timestamp']
        
        # Generate sample data
        data = []
        for i in range(rows):
            row = {
                'id': i + 1,
                'name': f'Item_{i+1}',
                'value': round(100 + (i * 0.5), 2),
                'category': f'Category_{(i % 5) + 1}',
                'timestamp': f'2025-01-{(i % 28) + 1:02d} 10:00:00'
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        file_path = Path(self.temp_dir) / filename
        df.to_csv(file_path, index=False)
        self.test_files.append(file_path)
        
        return file_path
    
    def create_test_json(self, filename: str, data: Dict[str, Any] = None) -> Path:
        """Create test JSON file"""
        if not self.temp_dir:
            self.setup_temp_directory()
            
        default_data = {
            "users": [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
            ],
            "metadata": {
                "version": "1.0",
                "created": "2025-09-23",
                "total_records": 2
            }
        }
        
        file_path = Path(self.temp_dir) / filename
        with open(file_path, 'w') as f:
            json.dump(data or default_data, f, indent=2)
        
        self.test_files.append(file_path)
        return file_path
    
    def create_test_jsonl(self, filename: str, records: int = 50) -> Path:
        """Create test JSONL file"""
        if not self.temp_dir:
            self.setup_temp_directory()
            
        file_path = Path(self.temp_dir) / filename
        with open(file_path, 'w') as f:
            for i in range(records):
                record = {
                    "id": i + 1,
                    "message": f"Test message {i + 1}",
                    "timestamp": f"2025-09-23T10:{i%60:02d}:00Z",
                    "score": round(0.5 + (i * 0.01), 3)
                }
                f.write(json.dumps(record) + '\n')
        
        self.test_files.append(file_path)
        return file_path
    
    def create_large_test_file(self, filename: str, size_mb: int = 50) -> Path:
        """Create large test file for performance testing"""
        if not self.temp_dir:
            self.setup_temp_directory()
            
        file_path = Path(self.temp_dir) / filename
        
        # Calculate approximate rows needed for target size
        rows_needed = size_mb * 1024 * 1024 // 100  # Assume ~100 bytes per row
        
        columns = ['id', 'data1', 'data2', 'data3', 'data4', 'description']
        data = []
        
        for i in range(rows_needed):
            row = {
                'id': i + 1,
                'data1': f'value_{i}',
                'data2': round(i * 0.123, 3),
                'data3': f'category_{i % 10}',
                'data4': i * 2,
                'description': f'This is a longer description for row {i} with some additional text to increase size'
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        self.test_files.append(file_path)
        
        return file_path
    
    def cleanup(self):
        """Clean up test files and directories"""
        try:
            # Clean up mock objects
            for mock_obj in self.mock_objects:
                if hasattr(mock_obj, 'reset_mock'):
                    mock_obj.reset_mock()
            
            # Remove test files
            for file_path in self.test_files:
                if file_path.exists():
                    file_path.unlink()
            
            # Remove temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

class MockFactory:
    """Factory for creating consistent mock objects"""
    
    @staticmethod
    def create_mock_llm_response(content: str = "Mock LLM response", 
                                model: str = "gpt-3.5-turbo",
                                token_count: int = 100) -> Dict[str, Any]:
        """Create mock LLM API response"""
        return {
            'content': content,
            'model': model,
            'token_count': token_count,
            'streaming': False,
            'raw_response': {
                'choices': [{'message': {'content': content}}],
                'usage': {'total_tokens': token_count}
            }
        }
    
    @staticmethod
    def create_mock_file_metadata(file_path: str, file_type: str = "csv", 
                                 size_bytes: int = 1024) -> Dict[str, Any]:
        """Create mock file metadata"""
        return {
            'path': file_path,
            'size_bytes': size_bytes,
            'file_type': file_type,
            'last_modified': time.time(),
            'checksum': 'mock_checksum_123',
            'columns': ['col1', 'col2', 'col3'] if file_type == 'csv' else [],
            'row_count': 100
        }
    
    @staticmethod
    def create_mock_cache_entry(value: Any, ttl: float = 3600.0) -> Mock:
        """Create mock cache entry"""
        mock_entry = Mock()
        mock_entry.value = value
        mock_entry.created_at = time.time()
        mock_entry.ttl = ttl
        mock_entry.hit_count = 0
        mock_entry.last_accessed = time.time()
        mock_entry.is_expired.return_value = False
        mock_entry.touch = Mock()
        return mock_entry
    
    @staticmethod
    def create_mock_query_profile(query: str = "test query", 
                                 complexity: str = "moderate") -> Mock:
        """Create mock query profile"""
        from src.backend.core.intelligent_query_engine import QueryType, QueryComplexity, AgentCapability
        
        mock_profile = Mock()
        mock_profile.query_text = query
        mock_profile.query_type = QueryType.DATA_ANALYSIS
        mock_profile.complexity = QueryComplexity.MODERATE
        mock_profile.estimated_duration = 5.0
        mock_profile.required_capabilities = {AgentCapability.STATISTICAL_ANALYSIS}
        mock_profile.expected_output_size = 2000
        mock_profile.priority = 5
        mock_profile.metadata = {'test': True}
        
        return mock_profile

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_data_manager():
    """Fixture providing test data manager"""
    manager = TestDataManager()
    yield manager
    manager.cleanup()

@pytest.fixture
def mock_factory():
    """Fixture providing mock factory"""
    return MockFactory()

@pytest.fixture
def sample_csv_file(test_data_manager):
    """Fixture providing sample CSV file"""
    return test_data_manager.create_test_csv("sample.csv", rows=50)

@pytest.fixture
def sample_json_file(test_data_manager):
    """Fixture providing sample JSON file"""
    return test_data_manager.create_test_json("sample.json")

@pytest.fixture
def sample_jsonl_file(test_data_manager):
    """Fixture providing sample JSONL file"""
    return test_data_manager.create_test_jsonl("sample.jsonl", records=30)

@pytest.fixture
def large_csv_file(test_data_manager):
    """Fixture providing large CSV file for performance testing"""
    return test_data_manager.create_large_test_file("large_sample.csv", size_mb=20)

@pytest.fixture
def mock_llm_client():
    """Fixture providing mock LLM client"""
    mock_client = AsyncMock()
    mock_client.analyze_batch = AsyncMock(return_value=[
        MockFactory.create_mock_llm_response("Test analysis result")
    ])
    mock_client.get_performance_metrics.return_value = {
        'total_requests': 100,
        'cache_hit_rate': 0.75,
        'average_duration': 2.5
    }
    return mock_client

@pytest.fixture
def mock_cache_manager():
    """Fixture providing mock cache manager"""
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)  # Default to cache miss
    mock_cache.put = AsyncMock()
    mock_cache.get_comprehensive_stats.return_value = {
        'overview': {'hit_rate': 75.5, 'total_requests': 1000}
    }
    return mock_cache

@pytest.fixture
def mock_file_processor():
    """Fixture providing mock file processor"""
    mock_processor = AsyncMock()
    mock_processor.process_file = AsyncMock(return_value={
        'file_type': 'csv',
        'row_count': 100,
        'columns': ['id', 'name', 'value'],
        'sample_data': [{'id': 1, 'name': 'test', 'value': 100}],
        'processing_time': 1.5
    })
    return mock_processor

# Performance measurement utilities
class PerformanceTimer:
    """Utility for measuring performance in tests"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        self.start_time = time.perf_counter()
        
    def stop(self):
        self.end_time = time.perf_counter()
        
    @property
    def elapsed(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

@pytest.fixture
def performance_timer():
    """Fixture providing performance timer"""
    return PerformanceTimer()

# Test markers for categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security
pytest.mark.fuzz = pytest.mark.fuzz

# Async test utilities
@asynccontextmanager
async def async_test_context():
    """Async context manager for tests"""
    try:
        yield
    finally:
        # Cleanup async resources
        await asyncio.sleep(0.01)  # Allow cleanup

# Test data constants
TEST_QUERIES = [
    "Analyze the sales data and show trends",
    "Create a visualization of the revenue by month",
    "What are the top 5 performing products?",
    "Calculate correlation between price and sales",
    "Predict next quarter's revenue",
    "Show me the customer segmentation analysis",
    "Generate financial summary report",
    "Identify outliers in the dataset"
]

TEST_FILE_SAMPLES = {
    'csv': {
        'small': (100, ['id', 'name', 'value']),
        'medium': (10000, ['id', 'category', 'price', 'quantity', 'date']),
        'large': (100000, ['id', 'data1', 'data2', 'data3', 'timestamp'])
    },
    'json': {
        'simple': {'key': 'value', 'number': 42},
        'complex': {
            'users': [{'id': i, 'name': f'User {i}'} for i in range(100)],
            'metadata': {'version': '1.0', 'count': 100}
        }
    }
}

# Error simulation utilities
class ErrorSimulator:
    """Utility for simulating various error conditions"""
    
    @staticmethod
    def network_error():
        """Simulate network error"""
        from aiohttp import ClientError
        raise ClientError("Simulated network error")
    
    @staticmethod
    def file_not_found_error():
        """Simulate file not found error"""
        raise FileNotFoundError("Simulated file not found")
    
    @staticmethod
    def memory_error():
        """Simulate memory error"""
        raise MemoryError("Simulated memory error")
    
    @staticmethod
    def timeout_error():
        """Simulate timeout error"""
        import asyncio
        raise asyncio.TimeoutError("Simulated timeout")

@pytest.fixture
def error_simulator():
    """Fixture providing error simulator"""
    return ErrorSimulator()

# Custom assertions for testing
def assert_performance_improvement(before_time: float, after_time: float, 
                                 min_improvement: float = 1.5):
    """Assert that performance improved by minimum factor"""
    improvement_factor = before_time / after_time
    assert improvement_factor >= min_improvement, \
        f"Performance improvement {improvement_factor:.2f}x is less than required {min_improvement}x"

def assert_memory_usage_reasonable(memory_mb: float, max_memory_mb: float = 100):
    """Assert that memory usage is within reasonable bounds"""
    assert memory_mb <= max_memory_mb, \
        f"Memory usage {memory_mb:.2f}MB exceeds limit {max_memory_mb}MB"

def assert_cache_hit_rate(hit_rate: float, min_hit_rate: float = 0.7):
    """Assert that cache hit rate meets minimum threshold"""
    assert hit_rate >= min_hit_rate, \
        f"Cache hit rate {hit_rate:.2f} is below minimum {min_hit_rate:.2f}"

# Test server fixtures
@pytest.fixture(scope="session")
def test_server_process():
    """Start test server for E2E tests"""
    # Mock server URL for testing
    return "http://localhost:8000"

@pytest.fixture
def multiple_browser_sessions():
    """Fixture providing multiple browser sessions for concurrent testing"""
    sessions = []
    try:
        # Mock multiple browser sessions
        for i in range(3):
            session = Mock()
            session.id = f"session_{i}"
            session.get = Mock()
            session.post = Mock()
            sessions.append(session)
        yield sessions
    finally:
        # Cleanup sessions
        for session in sessions:
            if hasattr(session, 'close'):
                session.close()

@pytest.fixture  
def browser_driver():
    """Fixture providing browser driver for E2E tests"""
    driver = Mock()
    driver.get = Mock()
    driver.find_element = Mock()
    driver.execute_script = Mock()
    driver.quit = Mock()
    yield driver
    driver.quit()

# Test configuration
TEST_CONFIG = {
    'timeout': 30,  # seconds
    'max_memory_mb': 500,
    'min_performance_improvement': 2.0,
    'min_cache_hit_rate': 0.75,
    'max_response_time': 10.0,
    'concurrent_users': 10
}