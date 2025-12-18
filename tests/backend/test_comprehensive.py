"""
Comprehensive test suite for Nexus LLM Analytics
Covers all major components and edge cases
"""

import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Add src directory to Python path
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from backend.core.sandbox import EnhancedSandbox
from backend.core.security_guards import SecurityGuards, CodeValidator
from backend.core.error_handling import (
    NexusError, ValidationError, DataProcessingError,
    ErrorHandler, handle_errors
)
from backend.core.rate_limiter import RateLimiter, TokenBucket
from backend.core.config import Settings, get_settings
from backend.core.chromadb_client import ChromaDBClient, chunk_text
from backend.core.model_selector import ModelSelector

# Test fixtures
@pytest.fixture
def sandbox():
    """Create sandbox instance for testing"""
    return EnhancedSandbox(max_memory_mb=128, max_cpu_seconds=10)

@pytest.fixture
def error_handler():
    """Create error handler instance"""
    return ErrorHandler()

@pytest.fixture
def rate_limiter():
    """Create rate limiter instance"""
    return RateLimiter(requests_per_minute=10, burst_size=5)

@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing"""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'score': [85.5, 92.0, 78.5]
    })

# Sandbox Tests
class TestSandbox:
    """Test suite for sandbox functionality"""
    
    def test_sandbox_safe_code_execution(self, sandbox, sample_dataframe):
        """Test safe code execution"""
        code = """
result = data.describe()
mean_age = data['age'].mean()
"""
        result = sandbox.execute(code, data=sample_dataframe)
        assert 'result' in result
        assert 'mean_age' in result['result']
        assert result['result']['mean_age'] == 30.0
    
    def test_sandbox_blocks_dangerous_imports(self, sandbox):
        """Test blocking of dangerous imports"""
        dangerous_code = """
import os
result = os.listdir('/')
"""
        result = sandbox.execute(dangerous_code)
        assert 'error' in result
        assert 'import' in result['error'].lower()
    
    def test_sandbox_blocks_file_operations(self, sandbox):
        """Test blocking of file operations"""
        dangerous_code = """
with open('/etc/passwd', 'r') as f:
    content = f.read()
"""
        result = sandbox.execute(dangerous_code)
        assert 'error' in result
    
    def test_sandbox_memory_limit(self, sandbox):
        """Test memory limit enforcement"""
        memory_bomb = """
data = []
for i in range(10**9):
    data.append('x' * 10000)
"""
        result = sandbox.execute(memory_bomb)
        assert 'error' in result
    
    def test_sandbox_timeout(self, sandbox):
        """Test execution timeout"""
        infinite_loop = """
while True:
    pass
"""
        result = sandbox.execute(infinite_loop)
        assert 'error' in result
        assert 'timeout' in result['error'].lower()

# Security Guards Tests
class TestSecurityGuards:
    """Test suite for security guards"""
    
    def test_validate_ast_safe_code(self):
        """Test AST validation for safe code"""
        safe_code = "result = 1 + 2"
        is_valid, msg = CodeValidator.validate_ast(safe_code)
        assert is_valid is True
    
    def test_validate_ast_dangerous_function(self):
        """Test AST validation blocks dangerous functions"""
        dangerous_code = "eval('__import__(\"os\").system(\"ls\")')"
        is_valid, msg = CodeValidator.validate_ast(dangerous_code)
        assert is_valid is False
        assert 'eval' in msg
    
    def test_validate_patterns_safe_code(self):
        """Test pattern validation for safe code"""
        safe_code = "df = pd.DataFrame({'a': [1, 2, 3]})"
        is_valid, msg = CodeValidator.validate_code_patterns(safe_code)
        assert is_valid is True
    
    def test_validate_patterns_dangerous_code(self):
        """Test pattern validation blocks dangerous patterns"""
        dangerous_code = "exec('import os; os.system(\"rm -rf /\")')"
        is_valid, msg = CodeValidator.validate_code_patterns(dangerous_code)
        assert is_valid is False
        assert 'exec(' in msg

# Error Handling Tests
class TestErrorHandling:
    """Test suite for error handling"""
    
    def test_validation_error(self):
        """Test validation error creation and formatting"""
        error = ValidationError("Invalid input", field="email")
        error_dict = error.to_dict()
        assert error_dict['error'] is not None
        assert 'validation' in error_dict['error_code']
    
    def test_data_processing_error(self):
        """Test data processing error"""
        error = DataProcessingError("Failed to process data", operation="aggregation")
        error_dict = error.to_dict()
        assert 'data_processing' in error_dict['error_code']
    
    def test_error_handler(self, error_handler):
        """Test error handler functionality"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = error_handler.handle_error(e, raise_error=False)
            assert 'error' in result
            assert result['error_code'] == 'system_medium'
    
    def test_error_statistics(self, error_handler):
        """Test error statistics tracking"""
        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                error_handler.handle_error(e, raise_error=False)
        
        stats = error_handler.get_error_statistics()
        assert stats['total_errors'] == 3
        assert 'ValueError' in stats['error_counts']
    
    @handle_errors(default_return={"status": "error"})
    def test_error_decorator(self):
        """Test error handling decorator"""
        raise RuntimeError("Test error")
    
    def test_decorator_execution(self):
        """Test that decorator handles errors properly"""
        result = self.test_error_decorator()
        assert result == {"status": "error"}

# Rate Limiter Tests
class TestRateLimiter:
    """Test suite for rate limiting"""
    
    @pytest.mark.asyncio
    async def test_token_bucket_basic(self):
        """Test token bucket basic functionality"""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Consume tokens
        allowed, wait = await bucket.consume(3)
        assert allowed is True
        assert wait == 0
        
        # Try to consume more than available
        allowed, wait = await bucket.consume(5)
        assert allowed is False
        assert wait > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self, rate_limiter):
        """Test rate limiter allows requests within limit"""
        for i in range(5):
            allowed, retry = await rate_limiter.check_rate_limit(identifier="user1")
            assert allowed is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess(self, rate_limiter):
        """Test rate limiter blocks excessive requests"""
        # Consume all tokens
        for i in range(5):
            await rate_limiter.check_rate_limit(identifier="user1")
        
        # Next request should be blocked
        allowed, retry = await rate_limiter.check_rate_limit(identifier="user1")
        assert allowed is False
        assert retry is not None and retry > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_per_user(self, rate_limiter):
        """Test per-user rate limiting"""
        # User 1 consumes tokens
        for i in range(5):
            await rate_limiter.check_rate_limit(identifier="user1")
        
        # User 2 should still be allowed
        allowed, retry = await rate_limiter.check_rate_limit(identifier="user2")
        assert allowed is True

# Configuration Tests
class TestConfiguration:
    """Test suite for configuration"""
    
    def test_settings_creation(self):
        """Test settings object creation"""
        settings = Settings()
        assert settings.app_name == "Nexus LLM Analytics"
        assert settings.max_file_size == 100 * 1024 * 1024
    
    def test_settings_from_env(self, monkeypatch):
        """Test settings loading from environment variables"""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("MAX_FILE_SIZE", "50000000")
        
        settings = Settings()
        assert settings.log_level == "DEBUG"
        assert settings.max_file_size == 50000000
    
    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from string"""
        settings = Settings(cors_allowed_origins="http://localhost:3000,http://localhost:3001")
        assert len(settings.cors_allowed_origins) == 2
        assert "http://localhost:3001" in settings.cors_allowed_origins
    
    def test_get_paths(self):
        """Test path generation methods"""
        settings = Settings()
        upload_path = settings.get_upload_path()
        assert upload_path.exists() or upload_path.parent.exists()
    
    def test_safe_dict(self):
        """Test safe dictionary with masked sensitive values"""
        settings = Settings(ollama_api_key="secret123")
        safe_dict = settings.get_safe_dict()
        assert safe_dict['ollama_api_key'] == "***MASKED***"

# ChromaDB Tests
class TestChromaDB:
    """Test suite for ChromaDB client"""
    
    def test_chunk_text(self):
        """Test text chunking functionality"""
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 100 for chunk in chunks)
    
    @patch('backend.core.chromadb_client.requests.post')
    def test_embed_text(self, mock_post):
        """Test text embedding"""
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        from backend.core.chromadb_client import embed_text
        embedding = embed_text("test text")
        assert embedding == [0.1, 0.2, 0.3]

# Model Selector Tests
class TestModelSelector:
    """Test suite for model selection"""
    
    @patch('backend.core.model_selector.psutil.virtual_memory')
    def test_model_selection_high_ram(self, mock_memory):
        """Test model selection with high RAM"""
        mock_memory.return_value = Mock(
            total=16 * 1024**3,
            available=8 * 1024**3,
            used=8 * 1024**3,
            percent=50
        )
        
        primary, review, embedding = ModelSelector.select_optimal_models()
        assert "llama3.1:8b" in primary
    
    @patch('backend.core.model_selector.psutil.virtual_memory')
    def test_model_selection_low_ram(self, mock_memory):
        """Test model selection with low RAM"""
        mock_memory.return_value = Mock(
            total=4 * 1024**3,
            available=1 * 1024**3,
            used=3 * 1024**3,
            percent=75
        )
        
        primary, review, embedding = ModelSelector.select_optimal_models()
        assert "phi3:mini" in primary
    
    def test_validate_model_compatibility(self):
        """Test model compatibility validation"""
        with patch('backend.core.model_selector.psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3
            )
            
            compatible, msg = ModelSelector.validate_model_compatibility("ollama/phi3:mini")
            assert compatible is True

# Integration Tests
class TestIntegration:
    """Integration tests for multiple components"""
    
    @pytest.mark.asyncio
    async def test_sandbox_with_error_handling(self, sandbox, error_handler):
        """Test sandbox with error handling integration"""
        dangerous_code = "import os"
        
        try:
            result = sandbox.execute(dangerous_code)
            if 'error' in result:
                raise ValidationError("Code validation failed", field="code")
        except Exception as e:
            error_result = error_handler.handle_error(e, raise_error=False)
            assert 'error' in error_result
            assert error_result['error_code'] == 'validation_low'
    
    @pytest.mark.asyncio
    async def test_rate_limiter_with_error_handling(self, rate_limiter, error_handler):
        """Test rate limiter with error handling"""
        from backend.core.rate_limiter import RateLimitExceeded
        
        # Exhaust rate limit
        for i in range(10):
            await rate_limiter.check_rate_limit(identifier="test_user")
        
        # Check if properly blocked
        allowed, retry = await rate_limiter.check_rate_limit(identifier="test_user")
        if not allowed:
            try:
                raise RateLimitExceeded("Rate limit exceeded", retry_after=retry)
            except Exception as e:
                error_result = error_handler.handle_error(e, raise_error=False)
                assert 'error' in error_result

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
