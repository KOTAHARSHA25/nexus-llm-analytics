"""
Tests for src/backend/core/utils.py

Tests JsonFormatter, log_data_version, setup_logging, friendly_error, and AgentRegistry.
"""
import pytest
import json
import logging
import os
from unittest.mock import patch, MagicMock, mock_open
from src.backend.core.utils import (
    JsonFormatter, log_data_version, setup_logging, friendly_error, AgentRegistry
)


class TestJsonFormatter:
    """Test JsonFormatter log formatter"""
    
    def test_basic_format(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            "test_logger", logging.INFO, "path.py", 10, "Test message", None, None
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["level"] == "INFO"
        assert data["name"] == "test_logger"
        assert data["message"] == "Test message"
        assert "time" in data
    
    def test_with_exception(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            "logger", logging.ERROR, "path.py", 10, "Error occurred", None, exc_info
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestLogDataVersion:
    """Test log_data_version audit logging"""
    
    def test_creates_audit_log(self):
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file):
            with patch('os.makedirs') as mock_makedirs:
                log_data_version("upload", "test.csv", {"rows": 100})
                
                mock_makedirs.assert_called_once()
                mock_file.assert_called_once()
                
                # Check that JSON was written
                written_data = mock_file().write.call_args[0][0]
                entry = json.loads(written_data.strip())
                
                assert entry["event"] == "upload"
                assert entry["filename"] == "test.csv"
                assert entry["details"]["rows"] == 100
                assert "timestamp" in entry


class TestSetupLogging:
    """Test setup_logging function"""
    
    def test_default_level(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('logging.FileHandler') as mock_fh:
                mock_fh.return_value = MagicMock()
                
                setup_logging("test.log")
                
                logger = logging.getLogger()
                assert logger.level == logging.INFO
    
    def test_custom_level(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=True):
            with patch('logging.FileHandler') as mock_fh:
                mock_fh.return_value = MagicMock()
                
                setup_logging("test.log")
                
                logger = logging.getLogger()
                assert logger.level == logging.DEBUG
    
    def test_handlers_setup(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('logging.FileHandler') as mock_fh:
                mock_file_handler = MagicMock()
                mock_fh.return_value = mock_file_handler
                
                setup_logging("test.log")
                
                mock_fh.assert_called_once_with("test.log")
                mock_file_handler.setFormatter.assert_called_once()


class TestFriendlyError:
    """Test friendly_error function"""
    
    def test_with_suggestion(self):
        result = friendly_error("Something went wrong", "Try again later")
        
        assert result["error"] == "Something went wrong"
        assert result["suggestion"] == "Try again later"
    
    def test_default_suggestion(self):
        result = friendly_error("Error occurred")
        
        assert result["error"] == "Error occurred"
        assert "check your input" in result["suggestion"].lower()


class TestAgentRegistry:
    """Test AgentRegistry class"""
    
    def test_register_agent(self):
        registry = AgentRegistry()
        mock_agent = MagicMock()
        
        registry.register("test_agent", mock_agent)
        
        assert "test_agent" in registry.registry
        assert registry.registry["test_agent"] is mock_agent
    
    def test_get_agent(self):
        registry = AgentRegistry()
        mock_agent = MagicMock()
        
        registry.register("agent1", mock_agent)
        
        result = registry.get("agent1")
        assert result is mock_agent
    
    def test_get_nonexistent_agent(self):
        registry = AgentRegistry()
        
        result = registry.get("nonexistent")
        assert result is None
